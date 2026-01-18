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
def test_caching_iterator_handles_backend_failure(self):
    """
        Test that when the backend fails, caching_iter does not continue trying
        to consume data, and rolls back the cache.
        """

    def faulty_backend():
        data = [b'a', b'b', b'c', b'Fail', b'd', b'e', b'f']
        for d in data:
            if d == b'Fail':
                raise exception.GlanceException('Backend failure')
            yield d

    def consume(image_id):
        caching_iter = self.cache.get_caching_iter(image_id, None, faulty_backend())
        list(caching_iter)
    image_id = '1'
    self.assertRaises(exception.GlanceException, consume, image_id)
    self.assertFalse(self.cache.is_cached(image_id))