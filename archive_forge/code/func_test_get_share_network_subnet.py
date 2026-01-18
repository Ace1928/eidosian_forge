import ddt
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
def test_get_share_network_subnet(self):
    default_subnet = utils.get_default_subnet(self.user_client, self.sn['id'])
    subnet = self.user_client.get_share_network_subnet(self.sn['id'], default_subnet['id'])
    self.assertEqual(self.neutron_net_id, subnet['neutron_net_id'])
    self.assertEqual(self.neutron_subnet_id, subnet['neutron_subnet_id'])