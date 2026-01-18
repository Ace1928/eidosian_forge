import ddt
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
def test_get_invalid_share_network_subnet(self):
    self.assertRaises(exceptions.CommandFailed, self.user_client.get_share_network_subnet, self.sn['id'], 'invalid_subnet_id')