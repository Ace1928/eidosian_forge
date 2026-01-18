import datetime
import errno
import io
import os
import time
from unittest import mock
from oslo_utils import fileutils
from glance.image_cache.drivers import centralized_db
from glance.tests import functional
def test_centralized_db_worker_url_not_set(self):
    try:
        self.config(image_cache_driver='centralized_db')
        self.start_server(enable_cache=True, set_worker_url=False)
    except RuntimeError as e:
        expected_message = "'worker_self_reference_url' needs to be set if `centralized_db` is defined as cache driver for image_cache_driver config option."
        self.assertIn(expected_message, e.args)