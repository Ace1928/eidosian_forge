import time
from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def cache_queue(self, image_id, expected_code=202):
    path = '/v2/cache/%s' % image_id
    response = self.api_put(path)
    self.assertEqual(expected_code, response.status_code)