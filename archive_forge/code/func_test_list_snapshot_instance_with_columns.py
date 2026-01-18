import ddt
from oslo_utils import uuidutils
import testtools
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_list_snapshot_instance_with_columns(self):
    snapshot_instances = self.admin_client.list_snapshot_instances(self.snapshot['id'], columns='id,status')
    self.assertGreater(len(snapshot_instances), 0)
    expected_keys = ('Id', 'Status')
    unexpected_keys = ('Snapshot ID',)
    for si in snapshot_instances:
        for key in expected_keys:
            self.assertIn(key, si)
        for key in unexpected_keys:
            self.assertNotIn(key, si)
        self.assertTrue(uuidutils.is_uuid_like(si['Id']))