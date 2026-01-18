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
@cliutils.arg('share_server_id', metavar='<share_server_id>', help='ID of the share server to modify.')
@cliutils.arg('--state', metavar='<state>', default=constants.STATUS_ACTIVE, help='Indicate which state to assign the share server. Options include active, error, creating, deleting, managing, unmanaging, manage_error and unmanage_error. If no state is provided, active will be used.')
@api_versions.wraps('2.49')
def do_share_server_reset_state(cs, args):
    """Explicitly update the state of a share server (Admin only)."""
    cs.share_servers.reset_state(args.share_server_id, args.state)