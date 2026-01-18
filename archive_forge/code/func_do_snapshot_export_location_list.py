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
@api_versions.wraps('2.32')
@cliutils.arg('snapshot', metavar='<snapshot>', help='Name or ID of the snapshot.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id,path".')
def do_snapshot_export_location_list(cs, args):
    """List export locations of a given snapshot."""
    if args.columns is not None:
        list_of_keys = _split_columns(columns=args.columns)
    else:
        list_of_keys = ['ID', 'Path']
    snapshot = _find_share_snapshot(cs, args.snapshot)
    export_locations = cs.share_snapshot_export_locations.list(snapshot)
    cliutils.print_list(export_locations, list_of_keys)