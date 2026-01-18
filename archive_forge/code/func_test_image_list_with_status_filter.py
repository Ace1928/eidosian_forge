import uuid
import fixtures
from openstackclient.tests.functional.image import base
def test_image_list_with_status_filter(self):
    output = self.openstack('image list --status active', parse_output=True)
    self.assertIn('active', [img['Status'] for img in output])