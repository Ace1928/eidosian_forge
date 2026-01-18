from glance.api.middleware import cache_manage
from glance.api.v2 import cached_images
import glance.common.config
import glance.common.wsgi
import glance.image_cache
from glance.tests import utils as test_utils
from unittest import mock
import webob
def test_bogus_request(self):
    bogus_request = webob.Request.blank('/bogus/')
    resource = self.cache_manage_filter.process_request(bogus_request)
    self.assertIsNone(resource)