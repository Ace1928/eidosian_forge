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
@api_versions.wraps('2.26')
@cliutils.arg('share_network', metavar='<share-network>', help='Name or ID of share network to update.')
@cliutils.arg('--neutron-net-id', '--neutron-net_id', '--neutron_net_id', '--neutron_net-id', metavar='<neutron-net-id>', default=None, action='single_alias', help='Neutron network ID. Used to set up network for share servers. This option is deprecated for microversion >= 2.51.')
@cliutils.arg('--neutron-subnet-id', '--neutron-subnet_id', '--neutron_subnet_id', '--neutron_subnet-id', metavar='<neutron-subnet-id>', default=None, action='single_alias', help='Neutron subnet ID. Used to set up network for share servers. This subnet should belong to specified neutron network. This option is deprecated for microversion >= 2.51.')
@cliutils.arg('--name', metavar='<name>', default=None, help='Share network name.')
@cliutils.arg('--description', metavar='<description>', default=None, help='Share network description.')
def do_share_network_update(cs, args):
    """Update share network data."""
    values = {'neutron_net_id': args.neutron_net_id, 'neutron_subnet_id': args.neutron_subnet_id, 'name': args.name, 'description': args.description}
    share_network = _find_share_network(cs, args.share_network).update(**values)
    info = share_network._info.copy()
    cliutils.print_dict(info)