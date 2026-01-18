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
@cliutils.arg('snapshot', metavar='<snapshot>', help='Name or ID of the share snapshot to allow access to.')
@cliutils.arg('access_type', metavar='<access_type>', help='Access rule type (only "ip", "user"(user or group), "cert" or "cephx" are supported).')
@cliutils.arg('access_to', metavar='<access_to>', help='Value that defines access.')
def do_snapshot_access_allow(cs, args):
    """Allow read only access to a snapshot."""
    share_snapshot = _find_share_snapshot(cs, args.snapshot)
    access = share_snapshot.allow(args.access_type, args.access_to)
    cliutils.print_dict(access)