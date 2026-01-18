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
def test_connection_pool_limits_maximum_connections(self):
    conn1 = self.connection_pool.get_nowait()
    conn2 = self.connection_pool.get_nowait()
    self.assertRaises(queue.Empty, self.connection_pool.get_nowait)
    self.connection_pool.put_nowait(conn1)
    self.connection_pool.put_nowait(conn2)
    self.connection_pool.get_nowait()