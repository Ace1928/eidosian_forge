import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data('name', 'description')
def test_list_shares_by_inexact_option(self, option):
    shares = self.user_client.list_shares(filters={option + '~': option})
    self.assertGreaterEqual(len(shares), 3)
    self.assertTrue(any((self.private_share['id'] == s['ID'] for s in shares)))