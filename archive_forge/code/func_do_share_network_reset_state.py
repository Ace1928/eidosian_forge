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
@cliutils.arg('share_network', metavar='<share-network>', help='Share network name or ID.')
@cliutils.arg('--state', metavar='<state>', default=constants.STATUS_ACTIVE, help='Indicate which state to assign the share network. Options include active, error, network change. If no state is provided, active will be used.')
@api_versions.wraps('2.63')
def do_share_network_reset_state(cs, args):
    """Explicitly update the state of a share network (Admin only)."""
    share_network = _find_share_network(cs, args.share_network)
    cs.share_networks.reset_state(share_network, args.state)