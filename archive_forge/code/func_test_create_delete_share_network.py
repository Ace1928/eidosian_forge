import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.data({'name': data_utils.rand_name('autotest_share_network_name')}, {'description': 'fake_description'}, {'neutron_net_id': 'fake_neutron_net_id', 'neutron_subnet_id': 'fake_neutron_subnet_id'})
def test_create_delete_share_network(self, net_data):
    share_subnet_support = utils.share_network_subnets_are_supported()
    share_subnet_fields = ['neutron_net_id', 'neutron_subnet_id', 'availability_zone'] if share_subnet_support else []
    sn = self.create_share_network(cleanup_in_class=False, **net_data)
    default_subnet = utils.get_default_subnet(self.user_client, sn['id']) if share_subnet_support else None
    expected_data = {'name': 'None', 'description': 'None', 'neutron_net_id': 'None', 'neutron_subnet_id': 'None'}
    expected_data.update(net_data)
    share_network_expected_data = [(k, v) for k, v in expected_data.items() if k not in share_subnet_fields]
    share_subnet_expected_data = [(k, v) for k, v in expected_data.items() if k in share_subnet_fields]
    for k, v in share_network_expected_data:
        self.assertEqual(v, sn[k])
    for k, v in share_subnet_expected_data:
        self.assertEqual(v, default_subnet[k])
    self.admin_client.delete_share_network(sn['id'])
    self.admin_client.wait_for_share_network_deletion(sn['id'])