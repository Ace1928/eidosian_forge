from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_image_handle_update_deactivate(self):
    self.images.reactivate.return_value = None
    self.images.deactivate.return_value = None
    value = mock.MagicMock()
    image_id = '41f0e60c-ebb4-4375-a2b4-845ae8b9c995'
    value.id = image_id
    value.status = 'active'
    self.my_image.resource_id = image_id
    props = self.stack.t.t['resources']['my_image']['properties'].copy()
    props['active'] = False
    self.my_image.t = self.my_image.t.freeze(properties=props)
    prop_diff = {'active': False}
    self.my_image.reparse()
    self.images.update.return_value = value
    self.images.get.return_value = value
    self.my_image.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.images.deactivate.assert_called_once_with(self.my_image.resource_id)