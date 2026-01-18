from unittest import mock
from glance.api.v2 import cached_images
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_get_cache_state(self):
    self._main_test_helper(['get_cached_images,get_queued_images', 'get_cache_state', 'cache_list'], image_mock=False)