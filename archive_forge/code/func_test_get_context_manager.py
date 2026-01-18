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
def test_get_context_manager(self):
    self.assertThat(self.connection_pool.queue, matchers.HasLength(0))
    with self.connection_pool.acquire() as conn:
        self.assertEqual(1, self.connection_pool._acquired)
    self.assertEqual(0, self.connection_pool._acquired)
    self.assertThat(self.connection_pool.queue, matchers.HasLength(1))
    self.assertEqual(conn, self.connection_pool.queue[0].connection)