from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_check_is_image_mutable(self):
    context = mock.MagicMock()
    image = mock.MagicMock()
    context.is_admin = True
    context.owner = 'someuser'
    self.assertIsNone(policy.check_is_image_mutable(context, image))
    context.is_admin = False
    image.owner = None
    self.assertRaises(exception.Forbidden, policy.check_is_image_mutable, context, image)
    image.owner = 'someoneelse'
    self.assertRaises(exception.Forbidden, policy.check_is_image_mutable, context, image)
    image.owner = 'someoneelse'
    context.owner = None
    self.assertRaises(exception.Forbidden, policy.check_is_image_mutable, context, image)
    image.owner = 'someuser'
    context.owner = 'someuser'
    self.assertIsNone(policy.check_is_image_mutable(context, image))