from unittest import mock
import webob
from glance.api.v2 import cached_images
import glance.gateway
from glance import image_cache
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_cache_clear_invalid_target(self):
    self.config(image_cache_dir='fake_cache_directory')
    req = unit_test_utils.get_fake_request()
    req.headers.update({'x-image-cache-clear-target': 'invalid'})
    self.assertRaises(webob.exc.HTTPBadRequest, self.controller.clear_cache, req)