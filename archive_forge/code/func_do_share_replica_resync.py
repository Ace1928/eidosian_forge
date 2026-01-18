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
@cliutils.arg('replica', metavar='<replica>', help='ID of the share replica to resync.')
@api_versions.wraps('2.11')
def do_share_replica_resync(cs, args):
    """Attempt to update the share replica with its 'active' mirror."""
    replica = _find_share_replica(cs, args.replica)
    cs.share_replicas.resync(replica)