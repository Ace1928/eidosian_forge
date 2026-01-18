from openstack.block_storage.v3 import volume as _volume
from openstack.tests.functional.block_storage.v3 import base
def test_attachment(self):
    attachment = self.conn.block_storage.create_attachment(self.VOLUME_ID, connector={}, instance_id=self.server.id)
    self.assertIn('id', attachment)
    self.assertIn('status', attachment)
    self.assertIn('instance', attachment)
    self.assertIn('volume_id', attachment)
    self.assertIn('attached_at', attachment)
    self.assertIn('detached_at', attachment)
    self.assertIn('attach_mode', attachment)
    self.assertIn('connection_info', attachment)
    attachment = self.user_cloud.block_storage.delete_attachment(attachment.id, ignore_missing=False)