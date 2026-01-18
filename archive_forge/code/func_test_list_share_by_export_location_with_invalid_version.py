import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
def test_list_share_by_export_location_with_invalid_version(self):
    self.assertRaises(exceptions.CommandFailed, self.admin_client.list_shares, filters={'export_location': 'fake'}, microversion='2.34')