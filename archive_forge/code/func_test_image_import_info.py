from openstackclient.tests.functional.image import base
def test_image_import_info(self):
    output = self.openstack('image import info', parse_output=True)
    self.assertIsNotNone(output['import-methods'])