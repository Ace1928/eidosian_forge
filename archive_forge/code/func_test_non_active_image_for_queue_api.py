from unittest import mock
import webob
from glance.api.v2 import cached_images
import glance.gateway
from glance import image_cache
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_non_active_image_for_queue_api(self):
    self.config(image_cache_dir='fake_cache_directory')
    req = unit_test_utils.get_fake_request()
    for status in ('saving', 'queued', 'pending_delete', 'deactivated', 'importing', 'uploading'):
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            mock_get.return_value = FakeImage(status=status)
            self.assertRaises(webob.exc.HTTPBadRequest, self.controller.queue_image_from_api, req, image_id=UUID4)