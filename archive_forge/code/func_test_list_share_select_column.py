import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
def test_list_share_select_column(self):
    shares = self.user_client.list_shares(columns='Name,Size')
    self.assertTrue(any((s['Name'] is not None for s in shares)))
    self.assertTrue(any((s['Size'] is not None for s in shares)))
    self.assertTrue(all(('Description' not in s for s in shares)))