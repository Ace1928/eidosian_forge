import ddt
from oslo_utils import uuidutils
from manilaclient.tests.functional import base
def test_get_share_instance_export_location(self):
    self.skip_if_microversion_not_supported('2.14')
    client = self.admin_client
    share_instances = client.list_share_instances(self.share['id'])
    self.assertGreater(len(share_instances), 0)
    self.assertIn('ID', share_instances[0])
    self.assertTrue(uuidutils.is_uuid_like(share_instances[0]['ID']))
    share_instance_id = share_instances[0]['ID']
    export_locations = client.list_share_instance_export_locations(share_instance_id)
    el = client.get_share_instance_export_location(share_instance_id, export_locations[0]['ID'])
    expected_keys = ('path', 'updated_at', 'created_at', 'id', 'preferred', 'is_admin_only', 'share_instance_id')
    for key in expected_keys:
        self.assertIn(key, el)
    self.assertIn(el['is_admin_only'], ('True', 'False'))
    self.assertIn(el['preferred'], ('True', 'False'))
    self.assertTrue(uuidutils.is_uuid_like(el['id']))
    for list_k, get_k in (('ID', 'id'), ('Path', 'path'), ('Preferred', 'preferred'), ('Is Admin only', 'is_admin_only')):
        self.assertEqual(export_locations[0][list_k], el[get_k])