from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_image_handle_create(self):
    value = mock.MagicMock()
    image_id = '41f0e60c-ebb4-4375-a2b4-845ae8b9c995'
    value.id = image_id
    self.images.create.return_value = value
    self.image_tags.update.return_value = None
    props = self.stack.t.t['resources']['my_image']['properties'].copy()
    props['tags'] = ['tag1']
    props['extra_properties'] = {'hw_firmware_type': 'uefi'}
    self.my_image.t = self.my_image.t.freeze(properties=props)
    self.my_image.reparse()
    self.my_image.handle_create()
    self.assertEqual(image_id, self.my_image.resource_id)
    self.images.create.assert_called_once_with(architecture='test_architecture', container_format=u'bare', disk_format=u'qcow2', id=u'41f0e60c-ebb4-4375-a2b4-845ae8b9c995', kernel_id='12345678-1234-1234-1234-123456789012', os_distro='test_distro', ramdisk_id='12345678-1234-1234-1234-123456789012', visibility='private', min_disk=10, min_ram=512, name=u'cirros_image', protected=False, owner=u'test_owner', tags=['tag1'])
    self.images.update.assert_called_once_with(image_id, hw_firmware_type='uefi')