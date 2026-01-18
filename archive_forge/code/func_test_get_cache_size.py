import datetime
import errno
import io
import os
import time
from unittest import mock
from oslo_utils import fileutils
from glance.image_cache.drivers import centralized_db
from glance.tests import functional
def test_get_cache_size(self):
    self.start_server(enable_cache=True)
    images = self.load_data()
    self.driver = centralized_db.Driver()
    self.driver.configure()
    self.assertEqual(0, self.driver.get_cache_size())
    path = '/v2/cache/%s' % images['public']
    self.api_put(path)
    self.wait_for_caching(images['public'])
    self.assertEqual(len(DATA), self.driver.get_cache_size())