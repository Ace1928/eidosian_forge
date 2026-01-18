import filecmp
import os
import tempfile
from openstack.tests.functional import base
def test_create_image_update_properties(self):
    test_image = tempfile.NamedTemporaryFile(delete=False)
    test_image.write(b'\x00' * 1024 * 1024)
    test_image.close()
    image_name = self.getUniqueString('image')
    try:
        image = self.user_cloud.create_image(name=image_name, filename=test_image.name, disk_format='raw', container_format='bare', min_disk=10, min_ram=1024, wait=True)
        self.user_cloud.update_image_properties(image=image, name=image_name, foo='bar')
        image = self.user_cloud.get_image(image_name)
        self.assertIn('foo', image.properties)
        self.assertEqual(image.properties['foo'], 'bar')
    finally:
        self.user_cloud.delete_image(image_name, wait=True)