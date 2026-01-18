import collections
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import param_utils
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.engine import constraints as constr
from heat.rpc import api as rpc_api
def format_notification_body(stack):
    if stack.status is not None and stack.action is not None:
        state = '_'.join(stack.state)
    else:
        state = 'Unknown'
    updated_at = heat_timeutils.isotime(stack.updated_time)
    result = {rpc_api.NOTIFY_TENANT_ID: stack.context.tenant_id, rpc_api.NOTIFY_USER_ID: stack.context.username, rpc_api.NOTIFY_USERID: stack.context.user_id, rpc_api.NOTIFY_USERNAME: stack.context.username, rpc_api.NOTIFY_STACK_ID: stack.id, rpc_api.NOTIFY_STACK_NAME: stack.name, rpc_api.NOTIFY_STATE: state, rpc_api.NOTIFY_STATE_REASON: stack.status_reason, rpc_api.NOTIFY_CREATE_AT: heat_timeutils.isotime(stack.created_time), rpc_api.NOTIFY_TAGS: stack.tags, rpc_api.NOTIFY_UPDATE_AT: updated_at}
    if stack.t is not None:
        result[rpc_api.NOTIFY_DESCRIPTION] = stack.t[stack.t.DESCRIPTION]
    return result