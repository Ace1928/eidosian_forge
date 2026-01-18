import filecmp
import os
import tempfile
from openstack.tests.functional import base
def test_create_image_without_filename(self):
    image_name = self.getUniqueString('image')
    image = self.user_cloud.create_image(name=image_name, disk_format='raw', container_format='bare', min_disk=10, min_ram=1024, allow_duplicates=True, wait=False)
    self.assertEqual(image_name, image.name)
    self.user_cloud.delete_image(image.id, wait=True)