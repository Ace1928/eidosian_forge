from glance.api.middleware import cache_manage
from glance.api.v2 import cached_images
import glance.common.config
import glance.common.wsgi
import glance.image_cache
from glance.tests import utils as test_utils
from unittest import mock
import webob
@mock.patch.object(cached_images.CacheController, 'get_cached_images')
def test_get_cached_images(self, mock_get_cached_images):
    mock_get_cached_images.return_value = self.stub_value
    request = webob.Request.blank('/v2/cached_images')
    resource = self.cache_manage_filter.process_request(request)
    mock_get_cached_images.assert_called_with(request)
    self.assertEqual('"' + self.stub_value + '"', resource.body.decode('utf-8'))