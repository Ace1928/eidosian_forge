import time
from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_cache_clear_queued_images(self):
    self.start_server(enable_cache=True)
    images = self.load_data()
    self.cache_queue(images['public'])
    self.cache_queue(images['private'])
    output = self.list_cache()
    self.assertEqual(2, len(output['queued_images']))
    self.assertEqual(0, len(output['cached_images']))
    self.cache_clear(target='queue')
    output = self.list_cache()
    self.assertEqual(0, len(output['queued_images']))
    self.assertEqual(0, len(output['cached_images']))