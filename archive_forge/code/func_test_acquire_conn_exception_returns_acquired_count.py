import queue
import threading
import time
from unittest import mock
import testtools
from testtools import matchers
from oslo_cache import _bmemcache_pool
from oslo_cache import _memcache_pool
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_acquire_conn_exception_returns_acquired_count(self):

    class TestException(Exception):
        pass
    with mock.patch.object(_TestConnectionPool, '_create_connection', side_effect=TestException):
        with testtools.ExpectedException(TestException):
            with self.connection_pool.acquire():
                pass
        self.assertThat(self.connection_pool.queue, matchers.HasLength(0))
        self.assertEqual(0, self.connection_pool._acquired)