import datetime
from oslo_utils import timeutils
from novaclient.tests.functional import base
def test_boot_server_using_image_with(self):
    """Scenario test which does the following:

        1. Create a server.
        2. Create a snapshot image of the server with a special meta key.
        3. Create a second server using the --image-with option using the meta
           key stored in the snapshot image created in step 2.
        """
    server_info = self.nova('boot', params='--flavor %(flavor)s --image %(image)s --poll image-with-server-1' % {'image': self.image.id, 'flavor': self.flavor.id})
    server_id = self._get_value_from_the_table(server_info, 'id')
    self.addCleanup(self._cleanup_server, server_id)
    snapshot_info = self.nova('image-create', params='--metadata image_with_meta=%(meta_value)s --show --poll %(server_id)s image-with-snapshot' % {'meta_value': server_id, 'server_id': server_id})
    snapshot_id = self._get_value_from_the_table(snapshot_info, 'id')
    self.addCleanup(self.glance.images.delete, snapshot_id)
    meta_value = self._get_value_from_the_table(snapshot_info, 'image_with_meta')
    self.assertEqual(server_id, meta_value)
    server_info = self.nova('boot', params='--flavor %(flavor)s --image-with image_with_meta=%(meta_value)s --poll image-with-server-2' % {'meta_value': server_id, 'flavor': self.flavor.id})
    server_id = self._get_value_from_the_table(server_info, 'id')
    self.addCleanup(self._cleanup_server, server_id)