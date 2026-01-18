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
@cliutils.arg('export_location', metavar='<export_location>', help='ID of the share instance export location.')
def do_share_instance_export_location_show(cs, args):
    """Show export location for the share instance."""
    instance = _find_share_instance(cs, args.instance)
    export_location = cs.share_instance_export_locations.get(instance, args.export_location)
    view_data = export_location._info.copy()
    cliutils.print_dict(view_data)