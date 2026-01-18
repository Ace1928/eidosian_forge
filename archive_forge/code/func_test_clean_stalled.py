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
@skip_if_disabled
def test_clean_stalled(self):
    """Test the clean method removes expected images."""
    incomplete_file_path = os.path.join(self.cache_dir, 'incomplete', '1')
    incomplete_file = open(incomplete_file_path, 'wb')
    incomplete_file.write(FIXTURE_DATA)
    incomplete_file.close()
    self.assertTrue(os.path.exists(incomplete_file_path))
    self.delay_inaccurate_clock()
    self.cache.clean(stall_time=0)
    self.assertFalse(os.path.exists(incomplete_file_path))