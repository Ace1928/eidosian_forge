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
@cliutils.arg('share_network', metavar='<share-network>', help='Name or ID of the share network to show.')
def do_share_network_show(cs, args):
    """Retrieve details for a share network."""
    share_network = _find_share_network(cs, args.share_network)
    info = share_network._info.copy()
    cliutils.print_dict(info)