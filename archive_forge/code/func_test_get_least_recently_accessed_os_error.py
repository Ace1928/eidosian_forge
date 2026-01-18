from contextlib import contextmanager
import datetime
import errno
import io
import os
import tempfile
import time
from unittest import mock
import fixtures
import glance_store as store
from oslo_config import cfg
from oslo_utils import fileutils
from oslo_utils import secretutils
from oslo_utils import units
from glance import async_
from glance.common import exception
from glance import context
from glance import gateway as glance_gateway
from glance import image_cache
from glance.image_cache import prefetcher
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
from glance.tests.utils import skip_if_disabled
from glance.tests.utils import xattr_writes_supported
def test_get_least_recently_accessed_os_error(self):
    self.assertEqual(0, self.cache.get_cache_size())
    for x in range(10):
        FIXTURE_FILE = io.BytesIO(FIXTURE_DATA)
        self.assertTrue(self.cache.cache_image_file(x, FIXTURE_FILE))
    self.assertEqual(10 * units.Ki, self.cache.get_cache_size())
    with mock.patch.object(os, 'stat') as mock_stat:
        mock_stat.side_effect = OSError
        image_id, size = self.cache.driver.get_least_recently_accessed()
        self.assertEqual(0, size)