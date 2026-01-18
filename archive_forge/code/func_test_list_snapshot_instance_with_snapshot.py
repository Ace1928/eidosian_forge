import ddt
from oslo_utils import uuidutils
import testtools
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_list_snapshot_instance_with_snapshot(self):
    snapshot_instances = self.admin_client.list_snapshot_instances(snapshot_id=self.snapshot['id'])
    self.assertEqual(1, len(snapshot_instances))
    expected_keys = ('ID', 'Snapshot ID', 'Status')
    for si in snapshot_instances:
        for key in expected_keys:
            self.assertIn(key, si)
        self.assertTrue(uuidutils.is_uuid_like(si['ID']))
        self.assertTrue(uuidutils.is_uuid_like(si['Snapshot ID']))