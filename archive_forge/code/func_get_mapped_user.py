import hashlib
from oslo_log import log
from keystone.auth import core
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils
from keystone.i18n import _
def get_mapped_user(self, project_id=None, domain_id=None):
    """Map client certificate to an existing user.

        If user is ephemeral, there is no validation on the user himself;
        however it will be mapped to a corresponding group(s) and the scope
        of this ephemeral user is the same as what is assigned to the group.

        :param project_id:  Project scope of the mapped user.
        :param domain_id: Domain scope of the mapped user.
        :returns: A dictionary that contains the keys, such as
            user_id, user_name, domain_id, domain_name
        :rtype: dict
        """
    idp_id = self._build_idp_id()
    LOG.debug('The IdP Id %s and protocol Id %s are used to look up the mapping.', idp_id, CONF.tokenless_auth.protocol)
    mapped_properties, mapping_id = self.federation_api.evaluate(idp_id, CONF.tokenless_auth.protocol, self.env)
    user = mapped_properties.get('user', {})
    user_id = user.get('id')
    user_name = user.get('name')
    user_type = user.get('type')
    if user.get('domain') is not None:
        user_domain_id = user.get('domain').get('id')
        user_domain_name = user.get('domain').get('name')
    else:
        user_domain_id = None
        user_domain_name = None
    if user_type == utils.UserType.EPHEMERAL:
        user_ref = {'type': utils.UserType.EPHEMERAL}
        group_ids = mapped_properties['group_ids']
        utils.validate_mapped_group_ids(group_ids, mapping_id, self.identity_api)
        group_ids.extend(utils.transform_to_group_ids(mapped_properties['group_names'], mapping_id, self.identity_api, self.resource_api))
        roles = self.assignment_api.get_roles_for_groups(group_ids, project_id, domain_id)
        if roles is not None:
            role_names = [role['name'] for role in roles]
            user_ref['roles'] = role_names
        user_ref['group_ids'] = list(group_ids)
        user_ref[federation_constants.IDENTITY_PROVIDER] = idp_id
        user_ref[federation_constants.PROTOCOL] = CONF.tokenless_auth.protocol
        return user_ref
    if user_id:
        user_ref = self.identity_api.get_user(user_id)
    elif user_name and (user_domain_name or user_domain_id):
        if user_domain_name:
            user_domain = self.resource_api.get_domain_by_name(user_domain_name)
            self.resource_api.assert_domain_enabled(user_domain['id'], user_domain)
            user_domain_id = user_domain['id']
        user_ref = self.identity_api.get_user_by_name(user_name, user_domain_id)
    else:
        msg = _('User auth cannot be built due to missing either user id, or user name with domain id, or user name with domain name.')
        raise exception.ValidationError(msg)
    self.identity_api.assert_user_enabled(user_id=user_ref['id'], user=user_ref)
    user_ref['type'] = utils.UserType.LOCAL
    return user_ref