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
@cliutils.arg('snapshot_instance', metavar='<snapshot_instance>', help='ID of the share snapshot instance.')
@cliutils.arg('export_location', metavar='<export_location>', help='ID of the share snapshot instance export location.')
def do_snapshot_instance_export_location_show(cs, args):
    """Show export location of the share instance snapshot."""
    snapshot_instance = _find_share_snapshot_instance(cs, args.snapshot_instance)
    export_location = cs.share_snapshot_instance_export_locations.get(args.export_location, snapshot_instance)
    view_data = export_location._info.copy()
    cliutils.print_dict(view_data)