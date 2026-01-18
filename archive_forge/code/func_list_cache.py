import datetime
import errno
import io
import os
import time
from unittest import mock
from oslo_utils import fileutils
from glance.image_cache.drivers import centralized_db
from glance.tests import functional
def list_cache(self, expected_code=200):
    path = '/v2/cache'
    response = self.api_get(path)
    self.assertEqual(expected_code, response.status_code)
    if response.status_code == 200:
        return response.json