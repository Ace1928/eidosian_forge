import ddt
from oslo_utils import uuidutils
import testtools
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.data('admin', 'user')
def test_list_snapshot_export_locations(self, role):
    client = self.admin_client if role == 'admin' else self.user_client
    export_locations = client.list_snapshot_export_locations(self.snapshot['id'])
    self.assertGreater(len(export_locations), 0)
    expected_keys = ('ID', 'Path')
    for el in export_locations:
        for key in expected_keys:
            self.assertIn(key, el)
        self.assertTrue(uuidutils.is_uuid_like(el['ID']))