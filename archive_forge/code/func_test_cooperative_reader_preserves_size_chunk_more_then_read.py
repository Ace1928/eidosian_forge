import io
import tempfile
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from oslo_config import cfg
from oslo_log import log as logging
import webob
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
from glance.tests.unit import base
from glance.tests import utils as test_utils
def test_cooperative_reader_preserves_size_chunk_more_then_read(self):
    chunk_size = 16 * 1024 * 1024
    read_size = 8 * 1024
    self._test_reader_chunked(chunk_size, read_size)