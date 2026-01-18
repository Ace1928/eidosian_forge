import ddt
from oslo_utils import uuidutils
import testtools
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_get_snapshot_instance_export_location(self):
    client = self.admin_client
    snapshot_instances = client.list_snapshot_instances(self.snapshot['id'])
    self.assertGreater(len(snapshot_instances), 0)
    self.assertIn('ID', snapshot_instances[0])
    self.assertTrue(uuidutils.is_uuid_like(snapshot_instances[0]['ID']))
    snapshot_instance_id = snapshot_instances[0]['ID']
    export_locations = client.list_snapshot_instance_export_locations(snapshot_instance_id)
    el = client.get_snapshot_instance_export_location(snapshot_instance_id, export_locations[0]['ID'])
    expected_keys = ['path', 'id', 'is_admin_only', 'share_snapshot_instance_id', 'updated_at', 'created_at']
    for key in expected_keys:
        self.assertIn(key, el)
    for key, key_el in (('ID', 'id'), ('Path', 'path'), ('Is Admin only', 'is_admin_only')):
        self.assertEqual(export_locations[0][key], el[key_el])
    self.assertTrue(uuidutils.is_uuid_like(el['share_snapshot_instance_id']))
    self.assertTrue(uuidutils.is_uuid_like(el['id']))
    self.assertIn(el['is_admin_only'], ('True', 'False'))