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
def test_get_caching_iter_when_open_fails(self):

    class OpenFailingDriver(object):

        def is_cacheable(self, *args, **kwargs):
            return True

        @contextmanager
        def open_for_write(self, *args, **kwargs):
            raise IOError
    self.driver = OpenFailingDriver()
    cache = image_cache.ImageCache()
    data = [b'a', b'b', b'c', b'd', b'e', b'f']
    caching_iter = cache.get_caching_iter('dummy_id', None, iter(data))
    self.assertEqual(data, list(caching_iter))