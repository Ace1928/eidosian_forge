import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data('ID', 'Path')
def test_list_shares_by_export_location(self, option):
    export_locations = self.admin_client.list_share_export_locations(self.public_share['id'])
    shares = self.admin_client.list_shares(filters={'export_location': export_locations[0][option]})
    self.assertEqual(1, len(shares))
    self.assertTrue(any((self.public_share['id'] == s['ID'] for s in shares)))
    for share in shares:
        get = self.admin_client.get_share(share['ID'])
        self.assertEqual(self.public_name, get['name'])