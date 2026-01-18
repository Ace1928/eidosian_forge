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
def share_network_subnet_create_check(self, share_network_id, neutron_net_id=None, neutron_subnet_id=None, availability_zone=None, reset=False, microversion=None):
    params = self._combine_share_network_subnet_data(neutron_net_id=neutron_net_id, neutron_subnet_id=neutron_subnet_id, availability_zone=availability_zone)
    cmd = 'share-network-subnet-create-check %(network_id)s %(params)s' % {'network_id': share_network_id, 'params': params}
    if reset:
        cmd += '--reset %s' % reset
    return output_parser.details(self.manila(cmd, microversion=microversion))