import logging
import sys
from oslo_serialization import jsonutils
from oslo_utils import strutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import deployment_utils
from heatclient.common import event_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_format
from heatclient.common import template_utils
from heatclient.common import utils
import heatclient.exc as exc
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of the stack these resources belong to.'))
@utils.arg('--pre-create', action='store_true', default=False, help=_('Clear the pre-create hooks (optional)'))
@utils.arg('--pre-update', action='store_true', default=False, help=_('Clear the pre-update hooks (optional)'))
@utils.arg('--pre-delete', action='store_true', default=False, help=_('Clear the pre-delete hooks (optional)'))
@utils.arg('hook', metavar='<RESOURCE>', nargs='+', help=_('Resource names with hooks to clear. Resources in nested stacks can be set using slash as a separator: nested_stack/another/my_resource. You can use wildcards to match multiple stacks or resources: nested_stack/an*/*_resource'))
def do_hook_clear(hc, args):
    """Clear hooks on a given stack."""
    show_deprecated('heat hook-clear', 'openstack stack hook clear')
    if args.pre_create:
        hook_type = 'pre-create'
    elif args.pre_update:
        hook_type = 'pre-update'
    elif args.pre_delete:
        hook_type = 'pre-delete'
    else:
        hook_type = hook_utils.get_hook_type_via_status(hc, args.id)
    for hook_string in args.hook:
        hook = [b for b in hook_string.split('/') if b]
        resource_pattern = hook[-1]
        stack_id = args.id
        hook_utils.clear_wildcard_hooks(hc, stack_id, hook[:-1], hook_type, resource_pattern)