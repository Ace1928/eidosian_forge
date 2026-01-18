import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
@api_versions.wraps('3.3')
@utils.arg('--marker', metavar='<marker>', default=None, start_version='3.5', help='Begin returning message that appear later in the message list than that represented by this id. Default=None.')
@utils.arg('--limit', metavar='<limit>', default=None, start_version='3.5', help='Maximum number of messages to return. Default=None.')
@utils.arg('--sort', metavar='<key>[:<direction>]', default=None, start_version='3.5', help='Comma-separated list of sort keys and directions in the form of <key>[:<asc|desc>]. Valid keys: %s. Default=None.' % ', '.join(base.SORT_KEY_VALUES))
@utils.arg('--resource_uuid', metavar='<resource_uuid>', default=None, help='Filters results by a resource uuid. Default=None. %s' % FILTER_DEPRECATED)
@utils.arg('--resource_type', metavar='<type>', default=None, help='Filters results by a resource type. Default=None. %s' % FILTER_DEPRECATED)
@utils.arg('--event_id', metavar='<id>', default=None, help='Filters results by event id. Default=None. %s' % FILTER_DEPRECATED)
@utils.arg('--request_id', metavar='<request_id>', default=None, help='Filters results by request id. Default=None. %s' % FILTER_DEPRECATED)
@utils.arg('--level', metavar='<level>', default=None, help='Filters results by the message level. Default=None. %s' % FILTER_DEPRECATED)
@utils.arg('--filters', action=AppendFilters, type=str, nargs='*', start_version='3.33', metavar='<key=value>', default=None, help="Filter key and value pairs. Please use 'cinder list-filters' to check enabled filters from server. Use 'key~=value' for inexact filtering if the key supports. Default=None.")
def do_message_list(cs, args):
    """Lists all messages."""
    search_opts = {'resource_uuid': args.resource_uuid, 'event_id': args.event_id, 'request_id': args.request_id}
    if AppendFilters.filters:
        search_opts.update(shell_utils.extract_filters(AppendFilters.filters))
    if args.resource_type:
        search_opts['resource_type'] = args.resource_type.upper()
    if args.level:
        search_opts['message_level'] = args.level.upper()
    marker = args.marker if hasattr(args, 'marker') else None
    limit = args.limit if hasattr(args, 'limit') else None
    sort = args.sort if hasattr(args, 'sort') else None
    messages = cs.messages.list(search_opts=search_opts, marker=marker, limit=limit, sort=sort)
    columns = ['ID', 'Resource Type', 'Resource UUID', 'Event ID', 'User Message']
    if sort:
        sortby_index = None
    else:
        sortby_index = 0
    shell_utils.print_list(messages, columns, sortby_index=sortby_index)
    AppendFilters.filters = []