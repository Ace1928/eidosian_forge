import uuid
import fixtures
from openstackclient.tests.functional.image import base
def test_image_list_with_name_filter(self):
    output = self.openstack('image list --name ' + self.name, parse_output=True)
    self.assertIn(self.name, [img['Name'] for img in output])