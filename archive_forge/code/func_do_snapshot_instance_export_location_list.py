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
@cliutils.arg('instance', metavar='<instance>', help='Name or ID of the snapshot instance.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id,path,is_admin_only".')
def do_snapshot_instance_export_location_list(cs, args):
    """List export locations of a given snapshot instance."""
    if args.columns is not None:
        list_of_keys = _split_columns(columns=args.columns)
    else:
        list_of_keys = ['ID', 'Path', 'Is Admin only']
    instance = _find_share_snapshot_instance(cs, args.instance)
    export_locations = cs.share_snapshot_instance_export_locations.list(instance)
    cliutils.print_list(export_locations, list_of_keys)