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
@cliutils.arg('--all-tenants', '--all-projects', action='single_alias', dest='all_projects', metavar='<0|1>', nargs='?', type=int, const=1, default=0, help='Display information from all projects (Admin only).')
@cliutils.arg('--name', metavar='<name>', default=None, help='Filter results by name.')
@cliutils.arg('--status', metavar='<status>', default=None, help='Filter results by status.')
@cliutils.arg('--share-group-id', '--share_group_id', metavar='<share_group_id>', default=None, action='single_alias', help='Filter results by share group ID.')
@cliutils.arg('--limit', metavar='<limit>', type=int, default=None, help='Maximum number of share group snapshots to return. (Default=None)')
@cliutils.arg('--offset', metavar='<offset>', default=None, help='Start position of share group snapshot listing.')
@cliutils.arg('--sort-key', '--sort_key', metavar='<sort_key>', type=str, default=None, action='single_alias', help='Key to be sorted, available keys are %(keys)s. Default=None.' % {'keys': constants.SHARE_GROUP_SNAPSHOT_SORT_KEY_VALUES})
@cliutils.arg('--sort-dir', '--sort_dir', metavar='<sort_dir>', type=str, default=None, action='single_alias', help='Sort direction, available values are %(values)s. OPTIONAL: Default=None.' % {'values': constants.SORT_DIR_VALUES})
@cliutils.arg('--detailed', dest='detailed', default=True, help='Show detailed information about share group snapshots.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id,name".')
@cliutils.service_type('sharev2')
def do_share_group_snapshot_list(cs, args):
    """List share group snapshots with filters."""
    if args.columns is not None:
        list_of_keys = _split_columns(columns=args.columns)
    else:
        list_of_keys = ('id', 'name', 'status', 'description')
    all_projects = int(os.environ.get('ALL_TENANTS', os.environ.get('ALL_PROJECTS', args.all_projects)))
    search_opts = {'offset': args.offset, 'limit': args.limit, 'all_tenants': all_projects, 'name': args.name, 'status': args.status, 'share_group_id': args.share_group_id}
    share_group_snapshots = cs.share_group_snapshots.list(detailed=args.detailed, search_opts=search_opts, sort_key=args.sort_key, sort_dir=args.sort_dir)
    cliutils.print_list(share_group_snapshots, fields=list_of_keys, sortby_index=None)