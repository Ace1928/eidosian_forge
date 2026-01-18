import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.data(True, False)
def test_list_share_networks(self, all_tenants):
    share_networks = self.admin_client.list_share_networks(all_tenants)
    self.assertTrue(any((self.sn['id'] == sn['id'] for sn in share_networks)))
    for sn in share_networks:
        self.assertEqual(2, len(sn))
        self.assertIn('id', sn)
        self.assertIn('name', sn)