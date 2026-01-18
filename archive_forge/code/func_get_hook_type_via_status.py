import fnmatch
import logging
from heatclient._i18n import _
from heatclient import exc
def get_hook_type_via_status(hc, stack_id):
    try:
        stack = hc.stacks.get(stack_id=stack_id)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Stack not found: %s') % stack_id)
    else:
        if 'IN_PROGRESS' not in stack.stack_status:
            raise exc.CommandError(_('Stack status %s not IN_PROGRESS') % stack.stack_status)
    if 'CREATE' in stack.stack_status:
        hook_type = 'pre-create'
    elif 'UPDATE' in stack.stack_status:
        hook_type = 'pre-update'
    elif 'DELETE' in stack.stack_status:
        hook_type = 'pre-delete'
    else:
        raise exc.CommandError(_('Unexpected stack status %s, only create, update and delete supported') % stack.stack_status)
    return hook_type