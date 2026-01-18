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
def get_share_network_subnets(self, share_network, microversion=None):
    share_network = self.get_share_network(share_network, microversion=microversion)
    raw_subnets = share_network.get('share_network_subnets')
    subnets = ast.literal_eval(raw_subnets)
    for subnet in subnets:
        for k, v in subnet.items():
            subnet[k] = str(v) if v is None else v
    return subnets