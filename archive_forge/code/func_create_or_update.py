import socket
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.api.aws import exception
from heat.api.aws import utils as api_utils
from heat.common import exception as heat_exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.common import urlfetch
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
def create_or_update(self, req, action=None):
    """Implements CreateStack and UpdateStack API actions.

        Create or update stack as defined in template file.
        """

    def extract_args(params):
        """Extract request params and reformat them to match engine API.

            FIXME: we currently only support a subset of
            the AWS defined parameters (both here and in the engine)
            """
        keymap = {'TimeoutInMinutes': rpc_api.PARAM_TIMEOUT, 'DisableRollback': rpc_api.PARAM_DISABLE_ROLLBACK}
        if 'DisableRollback' in params and 'OnFailure' in params:
            msg = _('DisableRollback and OnFailure may not be used together')
            raise exception.HeatInvalidParameterCombinationError(detail=msg)
        result = {}
        for k in keymap:
            if k in params:
                result[keymap[k]] = params[k]
        if 'OnFailure' in params:
            value = params['OnFailure']
            if value == 'DO_NOTHING':
                result[rpc_api.PARAM_DISABLE_ROLLBACK] = 'true'
            elif value in ('ROLLBACK', 'DELETE'):
                result[rpc_api.PARAM_DISABLE_ROLLBACK] = 'false'
        return result
    if action not in self.CREATE_OR_UPDATE_ACTION:
        msg = _('Unexpected action %(action)s') % {'action': action}
        return exception.HeatInternalFailureError(detail=msg)
    engine_action = {self.CREATE_STACK: self.rpc_client.create_stack, self.UPDATE_STACK: self.rpc_client.update_stack}
    con = req.context
    stack_parms = self._extract_user_params(req.params)
    create_args = extract_args(req.params)
    try:
        templ = self._get_template(req)
    except socket.gaierror:
        msg = _('Invalid Template URL')
        return exception.HeatInvalidParameterValueError(detail=msg)
    if templ is None:
        msg = _('TemplateBody or TemplateUrl were not given.')
        return exception.HeatMissingParameterError(detail=msg)
    try:
        stack = template_format.parse(templ)
    except ValueError:
        msg = _('The Template must be a JSON or YAML document.')
        return exception.HeatInvalidParameterValueError(detail=msg)
    args = {'template': stack, 'params': stack_parms, 'files': {}, 'args': create_args}
    try:
        stack_name = req.params['StackName']
        if action == self.CREATE_STACK:
            args['stack_name'] = stack_name
        else:
            args['stack_identity'] = self._get_identity(con, stack_name)
        result = engine_action[action](con, **args)
    except Exception as ex:
        return exception.map_remote_error(ex)
    try:
        identity = identifier.HeatIdentifier(**result)
    except (ValueError, TypeError):
        response = result
    else:
        response = {'StackId': identity.arn()}
    return api_utils.format_response(action, response)