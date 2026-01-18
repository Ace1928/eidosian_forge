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
@cliutils.arg('replica', metavar='<replica>', help='ID of the share replica to modify.')
@cliutils.arg('--replica-state', '--replica_state', '--state', metavar='<replica_state>', default='out_of_sync', action='single_alias', help='Indicate which replica_state to assign the replica. Options include in_sync, out_of_sync, active, error. If no state is provided, out_of_sync will be used.')
@api_versions.wraps('2.11')
def do_share_replica_reset_replica_state(cs, args):
    """Explicitly update the 'replica_state' of a share replica."""
    replica = _find_share_replica(cs, args.replica)
    cs.share_replicas.reset_replica_state(replica, args.replica_state)