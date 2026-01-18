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
@cliutils.arg('snapshot', metavar='<snapshot>', help='Name or ID of the share snapshot to list access of.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "access_type,access_to".')
def do_snapshot_access_list(cs, args):
    """Show access list for a snapshot."""
    if args.columns is not None:
        list_of_keys = _split_columns(columns=args.columns)
    else:
        list_of_keys = ['id', 'access_type', 'access_to', 'state']
    snapshot = _find_share_snapshot(cs, args.snapshot)
    access_list = snapshot.access_list()
    cliutils.print_list(access_list, list_of_keys)