import time
from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_cache_delete(self):
    self.start_server(enable_cache=True)
    images = self.load_data()
    self.cache_queue(images['public'])
    self.wait_for_caching(images['public'])
    output = self.list_cache()
    self.assertEqual(1, len(output['cached_images']))
    self.assertIn(images['public'], output['cached_images'][0]['image_id'])
    self.cache_delete(images['public'])
    output = self.list_cache()
    self.assertEqual(0, len(output['cached_images']))