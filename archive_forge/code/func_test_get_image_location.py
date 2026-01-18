from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_get_image_location(self):
    self.policy.get_image_location()
    self.enforcer.enforce.assert_called_once_with(self.context, 'get_image_location', mock.ANY)