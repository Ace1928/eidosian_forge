import collections
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import param_utils
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.engine import constraints as constr
from heat.rpc import api as rpc_api
def format_stack_db_object(stack):
    """Return a summary representation of the given stack.

    Given a stack versioned DB object, return a representation of the given
    stack for a stack listing.
    """
    updated_time = heat_timeutils.isotime(stack.updated_at)
    created_time = heat_timeutils.isotime(stack.created_at)
    deleted_time = heat_timeutils.isotime(stack.deleted_at)
    tags = None
    if stack.tags:
        tags = [t.tag for t in stack.tags]
    info = {rpc_api.STACK_ID: dict(stack.identifier()), rpc_api.STACK_NAME: stack.name, rpc_api.STACK_DESCRIPTION: '', rpc_api.STACK_ACTION: stack.action, rpc_api.STACK_STATUS: stack.status, rpc_api.STACK_STATUS_DATA: stack.status_reason, rpc_api.STACK_CREATION_TIME: created_time, rpc_api.STACK_UPDATED_TIME: updated_time, rpc_api.STACK_DELETION_TIME: deleted_time, rpc_api.STACK_OWNER: stack.username, rpc_api.STACK_PARENT: stack.owner_id, rpc_api.STACK_USER_PROJECT_ID: stack.stack_user_project_id, rpc_api.STACK_TAGS: tags}
    return info