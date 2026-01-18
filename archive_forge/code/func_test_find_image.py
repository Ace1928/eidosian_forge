from openstack.tests.functional import base
from openstack.tests.functional.image.v2.test_image import TEST_IMAGE_NAME
def test_find_image(self):
    image = self._get_non_test_image()
    self.assertIsNotNone(image)
    sot = self.conn.compute.find_image(image.id)
    self.assertEqual(image.id, sot.id)
    self.assertEqual(image.name, sot.name)