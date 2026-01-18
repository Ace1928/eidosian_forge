from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_get_live_state_resource_is_deleted(self):
    self.my_image.resource_id = '1234'
    self.my_image.client().images.get.return_value = {'status': 'deleted'}
    self.assertRaises(exception.EntityNotFound, self.my_image.get_live_state, self.my_image.properties)