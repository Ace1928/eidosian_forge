from unittest import mock
from glance.api.v2 import cached_images
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_delete_cache_entry(self):
    self._main_test_helper(['delete_cached_image,delete_queued_image', 'delete_cache_entry', 'cache_delete', UUID1])