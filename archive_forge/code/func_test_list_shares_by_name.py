import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
def test_list_shares_by_name(self):
    shares = self.user_client.list_shares(filters={'name': self.private_name})
    self.assertEqual(1, len(shares))
    self.assertTrue(any((self.private_share['id'] == s['ID'] for s in shares)))
    for share in shares:
        get = self.user_client.get_share(share['ID'])
        self.assertEqual(self.private_name, get['name'])