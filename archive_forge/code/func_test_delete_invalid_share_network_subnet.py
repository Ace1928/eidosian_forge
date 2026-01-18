import ddt
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
def test_delete_invalid_share_network_subnet(self):
    self.assertRaises(exceptions.NotFound, self.user_client.delete_share_network_subnet, share_network_subnet='invalid_subnet_id', share_network=self.sn['id'])