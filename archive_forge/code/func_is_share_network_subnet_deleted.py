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
def is_share_network_subnet_deleted(self, share_network_subnet, share_network, microversion=None):
    """Says whether share network subnet is deleted or not.

        :param share_network_subnet: text -- Name or ID of share network subnet
        :param share_network: text -- Name or ID of share network the subnet
            belongs to
        """
    subnets = self.get_share_network_subnets(share_network)
    return not any((subnet['id'] == share_network_subnet for subnet in subnets))