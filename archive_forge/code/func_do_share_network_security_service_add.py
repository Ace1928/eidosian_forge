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
@cliutils.arg('security_service', metavar='<security-service>', help='Security service name or ID to associate with.')
def do_share_network_security_service_add(cs, args):
    """Associate security service with share network."""
    share_network = _find_share_network(cs, args.share_network)
    security_service = _find_security_service(cs, args.security_service)
    cs.share_networks.add_security_service(share_network, security_service)