from unittest import mock
from glance.api.v2 import cached_images
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_init_no_config(self):
    self.assertIsNone(cached_images.WORKER)
    self.config(image_cache_dir=None)
    cached_images.CacheController()
    self.assertIsNone(cached_images.WORKER)