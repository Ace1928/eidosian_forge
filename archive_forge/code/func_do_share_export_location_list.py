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
@api_versions.wraps('2.9')
@cliutils.arg('share', metavar='<share>', help='Name or ID of the share.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id,host,status".')
def do_share_export_location_list(cs, args):
    """List export locations of a given share."""
    if args.columns is not None:
        list_of_keys = _split_columns(columns=args.columns)
    else:
        list_of_keys = ['ID', 'Path', 'Preferred']
    share = _find_share(cs, args.share)
    export_locations = cs.share_export_locations.list(share)
    cliutils.print_list(export_locations, list_of_keys)