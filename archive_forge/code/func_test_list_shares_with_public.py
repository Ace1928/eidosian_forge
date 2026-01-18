import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data(True, False)
def test_list_shares_with_public(self, public):
    shares = self.user_client.list_shares(is_public=public)
    self.assertGreater(len(shares), 1)
    if public:
        self.assertTrue(all(('Project ID' in s for s in shares)))
    else:
        self.assertTrue(all(('Project ID' not in s for s in shares)))