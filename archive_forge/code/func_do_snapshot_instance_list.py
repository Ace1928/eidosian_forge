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
@api_versions.wraps('2.19')
@cliutils.arg('--snapshot', metavar='<snapshot>', default=None, help='Filter results by share snapshot ID.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id".')
@cliutils.arg('--detailed', metavar='<detailed>', default=False, help='Show detailed information about snapshot instances. (Default=False)')
def do_snapshot_instance_list(cs, args):
    """List share snapshot instances."""
    snapshot = _find_share_snapshot(cs, args.snapshot) if args.snapshot else None
    if args.columns is not None:
        list_of_keys = _split_columns(columns=args.columns)
    elif args.detailed:
        list_of_keys = ['ID', 'Snapshot ID', 'Status', 'Created_at', 'Updated_at', 'Share_id', 'Share_instance_id', 'Progress', 'Provider_location']
    else:
        list_of_keys = ['ID', 'Snapshot ID', 'Status']
    instances = cs.share_snapshot_instances.list(detailed=args.detailed, snapshot=snapshot)
    cliutils.print_list(instances, list_of_keys)