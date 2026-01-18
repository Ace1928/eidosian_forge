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
@cliutils.arg('instance', metavar='<instance>', help='Name or ID of the share instance.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id,host,status".')
def do_share_instance_export_location_list(cs, args):
    """List export locations of a given share instance."""
    if args.columns is not None:
        list_of_keys = _split_columns(columns=args.columns)
    else:
        list_of_keys = ['ID', 'Path', 'Is Admin only', 'Preferred']
    instance = _find_share_instance(cs, args.instance)
    export_locations = cs.share_instance_export_locations.list(instance)
    cliutils.print_list(export_locations, list_of_keys)