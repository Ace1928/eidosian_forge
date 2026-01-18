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
@cliutils.arg('snapshot_instance', metavar='<snapshot_instance>', help='ID of the snapshot instance to modify.')
@cliutils.arg('--state', metavar='<state>', default='available', help='Indicate which state to assign the snapshot instance. Options include available, error, creating, deleting, error_deleting. If no state is provided, available will be used.')
@api_versions.wraps('2.19')
def do_snapshot_instance_reset_state(cs, args):
    """Explicitly update the state of a share snapshot instance."""
    snapshot_instance = _find_share_snapshot_instance(cs, args.snapshot_instance)
    snapshot_instance.reset_state(args.state)