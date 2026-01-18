from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_image_get_live_state_v2(self):
    self.glanceclient.version = 2.0
    self.my_image.resource_id = '1234'
    images = mock.MagicMock()
    show_value = {'name': 'test', 'disk_format': 'qcow2', 'container_format': 'bare', 'active': None, 'protected': False, 'is_public': False, 'min_disk': 0, 'min_ram': 0, 'id': '41f0e60c-ebb4-4375-a2b4-845ae8b9c995', 'tags': [], 'architecture': 'test_architecture', 'kernel_id': '12345678-1234-1234-1234-123456789012', 'os_distro': 'test_distro', 'os_version': '1.0', 'owner': 'test_owner', 'ramdisk_id': '12345678-1234-1234-1234-123456789012', 'members': None, 'visibility': 'private'}
    image = show_value
    images.get.return_value = image
    self.my_image.client().images = images
    reality = self.my_image.get_live_state(self.my_image.properties)
    expected = {'name': 'test', 'disk_format': 'qcow2', 'container_format': 'bare', 'active': None, 'protected': False, 'min_disk': 0, 'min_ram': 0, 'id': '41f0e60c-ebb4-4375-a2b4-845ae8b9c995', 'tags': [], 'architecture': 'test_architecture', 'kernel_id': '12345678-1234-1234-1234-123456789012', 'os_distro': 'test_distro', 'os_version': '1.0', 'owner': 'test_owner', 'ramdisk_id': '12345678-1234-1234-1234-123456789012', 'members': None, 'visibility': 'private'}
    self.assertEqual(set(expected.keys()), set(reality.keys()))
    for key in expected:
        self.assertEqual(expected[key], reality[key])