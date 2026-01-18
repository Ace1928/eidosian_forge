import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data(1, 0)
def test_list_shares_for_all_tenants(self, all_tenants):
    shares = self.admin_client.list_shares(all_tenants=all_tenants)
    self.assertLessEqual(1, len(shares))
    if all_tenants:
        self.assertTrue(all(('Project ID' in s for s in shares)))
        for s_id in (self.private_share['id'], self.public_share['id'], self.admin_private_share['id']):
            self.assertTrue(any((s_id == s['ID'] for s in shares)))
    else:
        self.assertTrue(all(('Project ID' not in s for s in shares)))
        self.assertTrue(any((self.admin_private_share['id'] == s['ID'] for s in shares)))
        if self.private_share['project_id'] != self.admin_private_share['project_id']:
            for s_id in (self.private_share['id'], self.public_share['id']):
                self.assertFalse(any((s_id == s['ID'] for s in shares)))