import uuid
import fixtures
from openstackclient.tests.functional.image import base
def test_image_members(self):
    """Test member add, remove, accept"""
    output = self.openstack('token issue', parse_output=True)
    my_project_id = output['project_id']
    output = self.openstack('image show -f json ' + self.name, parse_output=True)
    if output['visibility'] == 'shared':
        self.openstack('image add project ' + self.name + ' ' + my_project_id)
        self.openstack('image set ' + '--accept ' + self.name)
        output = self.openstack('image list -f json ' + '--shared', parse_output=True)
        self.assertIn(self.name, [img['Name'] for img in output])
        self.openstack('image set ' + '--reject ' + self.name)
        output = self.openstack('image list -f json ' + '--shared', parse_output=True)
        self.openstack('image remove project ' + self.name + ' ' + my_project_id)