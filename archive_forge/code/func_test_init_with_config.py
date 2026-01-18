from unittest import mock
from glance.api.v2 import cached_images
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_init_with_config(self):
    self.assertIsNone(cached_images.WORKER)
    self.config(image_cache_dir='/tmp')
    cached_images.CacheController()
    self.assertIsNotNone(cached_images.WORKER)
    self.assertTrue(cached_images.WORKER.is_alive())
    cached_images.WORKER.terminate()