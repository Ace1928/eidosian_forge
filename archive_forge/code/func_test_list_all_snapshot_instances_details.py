import ddt
from oslo_utils import uuidutils
import testtools
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_list_all_snapshot_instances_details(self):
    snapshot_instances = self.admin_client.list_snapshot_instances(detailed=True)
    self.assertGreater(len(snapshot_instances), 0)
    expected_keys = ('ID', 'Snapshot ID', 'Status', 'Created_at', 'Updated_at', 'Share_id', 'Share_instance_id', 'Progress', 'Provider_location')
    for si in snapshot_instances:
        for key in expected_keys:
            self.assertIn(key, si)
        for key in ('ID', 'Snapshot ID', 'Share_id', 'Share_instance_id'):
            self.assertTrue(uuidutils.is_uuid_like(si[key]))