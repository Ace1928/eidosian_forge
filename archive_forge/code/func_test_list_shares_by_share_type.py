import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
def test_list_shares_by_share_type(self):
    share_type_id = self.user_client.get_share_type(self.private_share['share_type'])['ID']
    self._list_shares({'share_type': share_type_id})