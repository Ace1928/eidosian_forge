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
@api_versions.wraps('2.47')
@cliutils.arg('replica', metavar='<replica>', help='ID of the share replica.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id,path,replica_state".')
def do_share_replica_export_location_list(cs, args):
    """List export locations of a share replica."""
    if args.columns is not None:
        list_of_keys = _split_columns(columns=args.columns)
    else:
        list_of_keys = ['ID', 'Availability Zone', 'Replica State', 'Preferred', 'Path']
    replica = _find_share_replica(cs, args.replica)
    export_locations = cs.share_replica_export_locations.list(replica)
    cliutils.print_list(export_locations, list_of_keys)