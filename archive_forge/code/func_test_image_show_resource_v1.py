from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_image_show_resource_v1(self):
    self.glanceclient.version = 1.0
    self.my_image.resource_id = 'test_image_id'
    image = mock.MagicMock()
    images = mock.MagicMock()
    image.to_dict.return_value = {'image': 'info'}
    images.get.return_value = image
    self.my_image.client().images = images
    self.assertEqual({'image': 'info'}, self.my_image.FnGetAtt('show'))
    images.get.assert_called_once_with('test_image_id')