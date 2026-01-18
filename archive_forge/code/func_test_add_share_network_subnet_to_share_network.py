import ddt
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
def test_add_share_network_subnet_to_share_network(self):
    neutron_net_id = 'new_neutron_net_id'
    neutron_subnet_id = 'new_neutron_subnet_id'
    availability_zone = self._get_availability_zone()
    subnet = self.add_share_network_subnet(self.sn['id'], neutron_net_id, neutron_subnet_id, availability_zone, cleanup_in_class=False)
    self.assertEqual(neutron_net_id, subnet['neutron_net_id'])
    self.assertEqual(neutron_subnet_id, subnet['neutron_subnet_id'])
    self.assertEqual(availability_zone, subnet['availability_zone'])