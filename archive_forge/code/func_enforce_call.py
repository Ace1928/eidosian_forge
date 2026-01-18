import functools
import flask
from oslo_log import log
from oslo_policy import opts
from oslo_policy import policy as common_policy
from oslo_utils import strutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import policies
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
@classmethod
def enforce_call(cls, enforcer=None, action=None, target_attr=None, member_target_type=None, member_target=None, filters=None, build_target=None):
    """Enforce RBAC on the current request.

        This will do some legwork and then instantiate the Enforcer if an
        enforcer is not passed in.

        :param enforcer: A pre-instantiated Enforcer object (optional)
        :type enforcer: :class:`RBACEnforcer`
        :param action: the name of the rule/policy enforcement to be checked
                       against, e.g. `identity:get_user` (optional may be
                       replaced by decorating the method/function with
                       `policy_enforcer_action`.
        :type action: str
        :param target_attr: complete override of the target data. This will
                            replace all other generated target data meaning
                            `member_target_type` and `member_target` are
                            ignored. This will also prevent extraction of
                            data from the X-Subject-Token. The `target` dict
                            should contain a series of key-value pairs such
                            as `{'user': user_ref_dict}`.
        :type target_attr: dict
        :param member_target_type: the type of the target, e.g. 'user'. Both
                                   this and `member_target` must be passed if
                                   either is passed.
        :type member_target_type: str
        :param member_target: the (dict form) reference of the member object.
                              Both this and `member_target_type` must be passed
                              if either is passed.
        :type member_target: dict
        :param filters: A variable number of optional string filters, these are
                        used to extract values from the query params. The
                        filters are added to the request data that is passed to
                        the enforcer and may be used to determine policy
                        action. In practice these are mainly supplied in the
                        various "list" APIs and are un-used in the default
                        supplied policies.
        :type filters: iterable
        :param build_target: A function to build the target for enforcement.
                             This is explicitly done after authentication
                             in order to not leak existance data before
                             auth.
        :type build_target: function
        """
    policy_dict = {}
    action = action or getattr(flask.g, cls.ACTION_STORE_ATTR, None)
    if action not in _POSSIBLE_TARGET_ACTIONS:
        LOG.warning('RBAC: Unknown enforcement action name `%s`. Rejecting as Forbidden, this is a programming error and a bug should be filed with as much information about the request that caused this as possible.', action)
        raise exception.Forbidden(message=_('Internal RBAC enforcement error, invalid rule (action) name.'))
    setattr(flask.g, _ENFORCEMENT_CHECK_ATTR, True)
    cls._assert_is_authenticated()
    if cls._shared_admin_auth_token_set():
        LOG.warning('RBAC: Bypassing authorization')
        return
    policy_dict.update(flask.request.view_args)
    if target_attr is None and build_target is None:
        try:
            policy_dict.update(cls._extract_member_target_data(member_target_type, member_target))
        except exception.NotFound:
            LOG.debug('Extracting inferred target data resulted in "NOT FOUND (404)".')
            raise
        except Exception as e:
            LOG.warning('Unable to extract inferred target data during enforcement')
            LOG.debug(e, exc_info=True)
            raise exception.ForbiddenAction(action=action)
        subj_token_target_data = cls._extract_subject_token_target_data()
        if subj_token_target_data:
            policy_dict.setdefault('target', {}).update(subj_token_target_data)
    else:
        if target_attr and build_target:
            raise ValueError('Programming Error: A target_attr or build_target must be provided, but not both')
        policy_dict['target'] = target_attr or build_target()
    json_input = flask.request.get_json(force=True, silent=True) or {}
    policy_dict.update(json_input.copy())
    policy_dict.update(cls._extract_filter_values(filters))
    flattened = utils.flatten_dict(policy_dict)
    if LOG.logger.getEffectiveLevel() <= log.DEBUG:
        args_str = ', '.join(['%s=%s' % (k, v) for k, v in (flask.request.view_args or {}).items()])
        args_str = strutils.mask_password(args_str)
        LOG.debug('RBAC: Authorizing `%(action)s(%(args)s)`', {'action': action, 'args': args_str})
    ctxt = cls._get_oslo_req_context()
    enforcer_obj = enforcer or cls()
    enforcer_obj._enforce(credentials=ctxt, action=action, target=flattened)
    LOG.debug('RBAC: Authorization granted')