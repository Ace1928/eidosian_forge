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
def falloffend(image_id):
    data = [b'a', b'b', b'c', b'd', b'e', b'f']
    checksum = None
    caching_iter = self.cache.get_caching_iter(image_id, checksum, iter(data))
    self.assertEqual(b'a', next(caching_iter))