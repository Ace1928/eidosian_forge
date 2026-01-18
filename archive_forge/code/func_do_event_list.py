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
@utils.arg('id', metavar='<NAME or ID>', help=_('Name or ID of stack to show the events for.'))
@utils.arg('-r', '--resource', metavar='<RESOURCE>', help=_('Name of the resource to filter events by.'))
@utils.arg('-f', '--filters', metavar='<KEY1=VALUE1;KEY2=VALUE2...>', help=_('Filter parameters to apply on returned events. This can be specified multiple times, or once with parameters separated by a semicolon.'), action='append')
@utils.arg('-l', '--limit', metavar='<LIMIT>', help=_('Limit the number of events returned.'))
@utils.arg('-m', '--marker', metavar='<ID>', help=_('Only return events that appear after the given event ID.'))
@utils.arg('-n', '--nested-depth', metavar='<DEPTH>', help=_('Depth of nested stacks from which to display events. Note this cannot be specified with --resource.'))
@utils.arg('-F', '--format', metavar='<FORMAT>', help=_('The output value format, one of: log, table'), default='table')
def do_event_list(hc, args):
    """List events for a stack."""
    show_deprecated('heat event-list', 'openstack stack event list')
    display_fields = ['id', 'resource_status_reason', 'resource_status', 'event_time']
    event_args = {'resource_name': args.resource, 'limit': args.limit, 'marker': args.marker, 'filters': utils.format_parameters(args.filters), 'sort_dir': 'asc'}
    if args.nested_depth and args.resource:
        msg = _('--nested-depth cannot be specified with --resource')
        raise exc.CommandError(msg)
    if args.nested_depth:
        try:
            nested_depth = int(args.nested_depth)
        except ValueError:
            msg = _('--nested-depth invalid value %s') % args.nested_depth
            raise exc.CommandError(msg)
        del event_args['marker']
        del event_args['limit']
        display_fields.append('stack_name')
    else:
        nested_depth = 0
    events = event_utils.get_events(hc, stack_id=args.id, event_args=event_args, nested_depth=nested_depth, marker=args.marker, limit=args.limit)
    if len(events) >= 1:
        if hasattr(events[0], 'resource_name'):
            display_fields.insert(0, 'resource_name')
        else:
            display_fields.insert(0, 'logical_resource_id')
    if args.format == 'log':
        print(utils.event_log_formatter(events))
    else:
        utils.print_list(events, display_fields, sortby_index=None)