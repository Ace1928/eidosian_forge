import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@utils.skip_if_microversion_not_supported('2.51')
def test_create_delete_share_network_with_az(self):
    share_subnet_fields = ['neutron_net_id', 'neutron_subnet_id', 'availability_zone']
    az = self.user_client.list_availability_zones()[0]
    net_data = {'neutron_net_id': 'fake_neutron_net_id', 'neutron_subnet_id': 'fake_neutron_subnet_id', 'availability_zone': az['Name']}
    sn = self.create_share_network(cleanup_in_class=False, **net_data)
    default_subnet = utils.get_subnet_by_availability_zone_name(self.user_client, sn['id'], az['Name'])
    expected_data = {'name': 'None', 'description': 'None', 'neutron_net_id': 'None', 'neutron_subnet_id': 'None', 'availability_zone': 'None'}
    expected_data.update(net_data)
    share_network_expected_data = [(k, v) for k, v in expected_data.items() if k not in share_subnet_fields]
    share_subnet_expected_data = [(k, v) for k, v in expected_data.items() if k in share_subnet_fields]
    for k, v in share_network_expected_data:
        self.assertEqual(v, sn[k])
    for k, v in share_subnet_expected_data:
        self.assertEqual(v, default_subnet[k])
    self.admin_client.delete_share_network(sn['id'])
    self.admin_client.wait_for_share_network_deletion(sn['id'])