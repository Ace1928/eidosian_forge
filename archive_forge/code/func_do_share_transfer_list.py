from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
@api_versions.wraps('2.77')
@cliutils.arg('--all-tenants', '--all-projects', action='single_alias', dest='all_projects', metavar='<0|1>', nargs='?', type=int, const=1, default=0, help='Shows details for all tenants. (Admin only).')
@cliutils.arg('--name', metavar='<name>', default=None, action='single_alias', help='Transfer name. Default=None.')
@cliutils.arg('--id', metavar='<id>', default=None, action='single_alias', help='Transfer ID. Default=None.')
@cliutils.arg('--resource-type', '--resource_type', metavar='<resource_type>', default=None, action='single_alias', help='Transfer type, which can be share or network. Default=None.')
@cliutils.arg('--resource-id', '--resource_id', metavar='<resource_id>', default=None, action='single_alias', help='Transfer resource id. Default=None.')
@cliutils.arg('--source-project-id', '--source_project_id', metavar='<source_project_id>', default=None, action='single_alias', help='Transfer source project id. Default=None.')
@cliutils.arg('--limit', metavar='<limit>', type=int, default=None, help='Maximum number of messages to return. (Default=None)')
@cliutils.arg('--offset', metavar='<offset>', default=None, help='Start position of message listing.')
@cliutils.arg('--sort-key', '--sort_key', metavar='<sort_key>', type=str, default=None, action='single_alias', help='Key to be sorted, available keys are %(keys)s. Default=None.' % {'keys': constants.SHARE_TRANSFER_SORT_KEY_VALUES})
@cliutils.arg('--sort-dir', '--sort_dir', metavar='<sort_dir>', type=str, default=None, action='single_alias', help='Sort direction, available values are %(values)s. Optional: Default=None.' % {'values': constants.SORT_DIR_VALUES})
@cliutils.arg('--detailed', dest='detailed', metavar='<0|1>', nargs='?', type=int, const=1, default=0, help='Show detailed information about filtered share transfers.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id,resource_id".')
def do_share_transfer_list(cs, args):
    """Lists all transfers."""
    if args.columns is not None:
        list_of_keys = _split_columns(columns=args.columns)
    else:
        list_of_keys = ['ID', 'Name', 'Resource Type', 'Resource Id']
    if args.detailed:
        list_of_keys.extend(['Created At', 'Expires At', 'Source Project Id', 'Destination Project Id', 'Accepted'])
    all_projects = int(os.environ.get('ALL_TENANTS', os.environ.get('ALL_PROJECTS', args.all_projects)))
    search_opts = {'offset': args.offset, 'limit': args.limit, 'all_tenants': all_projects, 'id': args.id, 'name': args.name, 'resource_type': args.resource_type, 'resource_id': args.resource_id, 'source_project_id': args.source_project_id}
    share_transfers = cs.transfers.list(detailed=args.detailed, search_opts=search_opts, sort_key=args.sort_key, sort_dir=args.sort_dir)
    cliutils.print_list(share_transfers, fields=list_of_keys, sortby_index=None)