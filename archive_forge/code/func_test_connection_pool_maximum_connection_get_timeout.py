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
def test_connection_pool_maximum_connection_get_timeout(self):
    connection_pool = _TestConnectionPool(maxsize=1, unused_timeout=self.unused_timeout, conn_get_timeout=0)

    def _acquire_connection():
        with connection_pool.acquire():
            pass
    conn = connection_pool.get_nowait()
    self.assertRaises(exception.QueueEmpty, _acquire_connection)
    connection_pool.put_nowait(conn)
    _acquire_connection()