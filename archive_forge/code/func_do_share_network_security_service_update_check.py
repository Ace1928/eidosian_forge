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
@cliutils.arg('--reset', metavar='<True|False>', choices=['True', 'False'], required=False, default=False, help='Reset and start again the check operation.(Optional, Default=False)')
@api_versions.wraps('2.63')
def do_share_network_security_service_update_check(cs, args):
    """Check if a security service update on the share network is supported.

    This call can be repeated until a successful result is obtained.
    """
    share_network = _find_share_network(cs, args.share_network)
    current_security_service = _find_security_service(cs, args.current_security_service)
    new_security_service = _find_security_service(cs, args.new_security_service)
    share_network_update_check = cs.share_networks.update_share_network_security_service_check(share_network, current_security_service, new_security_service, reset_operation=args.reset)
    cliutils.print_dict(share_network_update_check[1])