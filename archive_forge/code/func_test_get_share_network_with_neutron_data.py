import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_get_share_network_with_neutron_data(self):
    get = self.admin_client.get_share_network(self.sn['id'])
    self.assertEqual(self.name, get['name'])
    self.assertEqual(self.description, get['description'])
    if not utils.share_network_subnets_are_supported():
        self.assertEqual(self.neutron_net_id, get['neutron_net_id'])
        self.assertEqual(self.neutron_subnet_id, get['neutron_subnet_id'])