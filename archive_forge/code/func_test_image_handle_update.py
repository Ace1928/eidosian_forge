from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_image_handle_update(self):
    self.my_image.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    prop_diff = {'architecture': 'test_architecture', 'kernel_id': '12345678-1234-1234-1234-123456789012', 'os_distro': 'test_distro', 'owner': 'test_owner', 'ramdisk_id': '12345678-1234-1234-1234-123456789012'}
    self.my_image.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.images.update.assert_called_with(self.my_image.resource_id, architecture='test_architecture', kernel_id='12345678-1234-1234-1234-123456789012', os_distro='test_distro', owner='test_owner', ramdisk_id='12345678-1234-1234-1234-123456789012')