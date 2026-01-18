from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_image_handle_update_image_props(self):
    self.my_image.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    props = self.stack.t.t['resources']['my_image']['properties'].copy()
    props['extra_properties'] = {'hw_firmware_type': 'uefi'}
    self.my_image.t = self.my_image.t.freeze(properties=props)
    self.my_image.reparse()
    prop_diff = {'extra_properties': {'os_secure_boot': 'required'}}
    self._handle_update_image_props(prop_diff)