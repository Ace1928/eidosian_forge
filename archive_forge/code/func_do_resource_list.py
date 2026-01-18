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
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of stack to show the resources for.'))
@utils.arg('-n', '--nested-depth', metavar='<DEPTH>', help=_('Depth of nested stacks from which to display resources.'))
@utils.arg('--with-detail', default=False, action='store_true', help=_('Enable detail information presented for each resource in resources list.'))
@utils.arg('-f', '--filter', metavar='<KEY=VALUE>', help=_('Filter parameters to apply on returned resources based on their name, status, type, action, id and physical_resource_id. This can be specified multiple times.'), action='append')
def do_resource_list(hc, args):
    """Show list of resources belonging to a stack."""
    show_deprecated('heat resource-list', 'openstack stack resource list')
    fields = {'stack_id': args.id, 'nested_depth': args.nested_depth, 'with_detail': args.with_detail, 'filters': utils.format_parameters(args.filter)}
    try:
        resources = hc.resources.list(**fields)
    except exc.HTTPNotFound:
        raise exc.CommandError(_('Stack not found: %s') % args.id)
    else:
        fields = ['physical_resource_id', 'resource_type', 'resource_status', 'updated_time']
        if len(resources) >= 1 and (not hasattr(resources[0], 'resource_name')):
            fields.insert(0, 'logical_resource_id')
        else:
            fields.insert(0, 'resource_name')
        if args.nested_depth or args.with_detail:
            fields.append('stack_name')
        utils.print_list(resources, fields, sortby_index=4)