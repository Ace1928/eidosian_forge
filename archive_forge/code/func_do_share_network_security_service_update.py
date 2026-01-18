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
@cliutils.arg('current_security_service', metavar='<current-security-service>', help='Current security service name or ID.')
@cliutils.arg('new_security_service', metavar='<new-security-service>', help='New security service name or ID.')
@api_versions.wraps('2.63')
def do_share_network_security_service_update(cs, args):
    """Update a current security service to a new security service."""
    share_network = _find_share_network(cs, args.share_network)
    current_security_service = _find_security_service(cs, args.current_security_service)
    new_security_service = _find_security_service(cs, args.new_security_service)
    cs.share_networks.update_share_network_security_service(share_network, current_security_service, new_security_service)