from unittest import mock
import webob
from glance.api.v2 import cached_images
import glance.gateway
from glance import image_cache
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_delete_cached_image_disabled(self):
    req = webob.Request.blank('')
    req.context = 'test'
    self.assertRaises(webob.exc.HTTPNotFound, self.controller.delete_cached_image, req, image_id='test')