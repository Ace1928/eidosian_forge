import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
def add_share_network_subnet(self, share_network, neutron_net_id=None, neutron_subnet_id=None, availability_zone=None, microversion=None):
    """Create new share network subnet for the given share network."""
    params = self._combine_share_network_subnet_data(neutron_net_id=neutron_net_id, neutron_subnet_id=neutron_subnet_id, availability_zone=availability_zone)
    share_network_subnet_raw = self.manila('share-network-subnet-create %(sn)s %(params)s' % {'sn': share_network, 'params': params}, microversion=microversion)
    share_network_subnet = output_parser.details(share_network_subnet_raw)
    return share_network_subnet