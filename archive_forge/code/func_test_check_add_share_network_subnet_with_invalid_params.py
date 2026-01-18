import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.data({'neutron_net_id': None, 'neutron_subnet_id': 'fake_subnet_id'}, {'neutron_net_id': 'fake_net_id', 'neutron_subnet_id': None}, {'availability_zone': 'invalid_availability_zone'})
def test_check_add_share_network_subnet_with_invalid_params(self, params):
    self.assertRaises(tempest_lib_exc.CommandFailed, self.user_client.share_network_subnet_create_check, self.sn['id'], **params)