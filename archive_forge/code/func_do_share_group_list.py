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
@cliutils.arg('--name', metavar='<name>', type=str, default=None, help='Filter results by name.')
@cliutils.arg('--description', metavar='<description>', type=str, default=None, help='Filter results by description. Available only for microversion >= 2.36.')
@cliutils.arg('--status', metavar='<status>', type=str, default=None, help='Filter results by status.')
@cliutils.arg('--share-server-id', '--share-server_id', '--share_server-id', '--share_server_id', metavar='<share_server_id>', type=str, default=None, action='single_alias', help='Filter results by share server ID (Admin only).')
@cliutils.arg('--share-group-type', '--share-group-type-id', '--share_group_type', '--share_group_type_id', metavar='<share_group_type>', type=str, default=None, action='single_alias', help='Filter results by a share group type ID or name that was used for share group creation.')
@cliutils.arg('--snapshot', metavar='<snapshot>', type=str, default=None, help='Filter results by share group snapshot name or ID that was used to create the share group.')
@cliutils.arg('--host', metavar='<host>', default=None, help='Filter results by host.')
@cliutils.arg('--share-network', '--share_network', metavar='<share_network>', type=str, default=None, action='single_alias', help='Filter results by share-network name or ID.')
@cliutils.arg('--project-id', '--project_id', metavar='<project_id>', type=str, default=None, action='single_alias', help="Filter results by project ID. Useful with set key '--all-projects'.")
@cliutils.arg('--limit', metavar='<limit>', type=int, default=None, help='Maximum number of share groups to return. (Default=None)')
@cliutils.arg('--offset', metavar='<offset>', default=None, help='Start position of share group listing.')
@cliutils.arg('--sort-key', '--sort_key', metavar='<sort_key>', type=str, default=None, action='single_alias', help='Key to be sorted, available keys are %(keys)s. Default=None.' % {'keys': constants.SHARE_GROUP_SORT_KEY_VALUES})
@cliutils.arg('--sort-dir', '--sort_dir', metavar='<sort_dir>', type=str, default=None, action='single_alias', help='Sort direction, available values are %(values)s. OPTIONAL: Default=None.' % {'values': constants.SORT_DIR_VALUES})
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id,name".')
@cliutils.arg('--name~', metavar='<name~>', type=str, default=None, help='Filter results matching a share group name pattern. Available only for microversion >= 2.36.')
@cliutils.arg('--description~', metavar='<description~>', type=str, default=None, help='Filter results matching a share group description pattern. Available only for microversion >= 2.36.')
@cliutils.service_type('sharev2')
def do_share_group_list(cs, args):
    """List share groups with filters."""
    if args.columns is not None:
        list_of_keys = _split_columns(columns=args.columns)
    else:
        list_of_keys = ('ID', 'Name', 'Status', 'Description')
    all_projects = int(os.environ.get('ALL_TENANTS', os.environ.get('ALL_PROJECTS', args.all_projects)))
    empty_obj = type('Empty', (object,), {'id': None})
    sg_type = _find_share_group_type(cs, args.share_group_type) if args.share_group_type else empty_obj
    snapshot = _find_share_snapshot(cs, args.snapshot) if args.snapshot else empty_obj
    share_network = _find_share_network(cs, args.share_network) if args.share_network else empty_obj
    search_opts = {'offset': args.offset, 'limit': args.limit, 'all_tenants': all_projects, 'name': args.name, 'status': args.status, 'share_server_id': args.share_server_id, 'share_group_type_id': sg_type.id, 'source_share_group_snapshot_id': snapshot.id, 'host': args.host, 'share_network_id': share_network.id, 'project_id': args.project_id}
    if cs.api_version.matches(api_versions.APIVersion('2.36'), api_versions.APIVersion()):
        search_opts['name~'] = getattr(args, 'name~')
        search_opts['description~'] = getattr(args, 'description~')
        search_opts['description'] = getattr(args, 'description')
    elif getattr(args, 'name~') or getattr(args, 'description~') or getattr(args, 'description'):
        raise exceptions.CommandError('Pattern based filtering (name~, description~ and description) is only available with manila API version >= 2.36')
    share_groups = cs.share_groups.list(search_opts=search_opts, sort_key=args.sort_key, sort_dir=args.sort_dir)
    cliutils.print_list(share_groups, fields=list_of_keys, sortby_index=None)