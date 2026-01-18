from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_delete_locations(self):
    self.policy.delete_locations()
    self.enforcer.enforce.assert_called_once_with(self.context, 'delete_image_location', mock.ANY)