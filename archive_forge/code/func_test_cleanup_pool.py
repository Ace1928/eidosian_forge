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
def test_cleanup_pool(self):
    self.test_get_context_manager()
    newtime = time.time() + self.unused_timeout * 2
    non_expired_connection = _memcache_pool._PoolItem(ttl=newtime * 2, connection=mock.MagicMock())
    self.connection_pool.queue.append(non_expired_connection)
    self.assertThat(self.connection_pool.queue, matchers.HasLength(2))
    with mock.patch.object(time, 'time', return_value=newtime):
        conn = self.connection_pool.queue[0].connection
        with self.connection_pool.acquire():
            pass
        conn.assert_has_calls([mock.call(self.connection_pool.destroyed_value)])
    self.assertThat(self.connection_pool.queue, matchers.HasLength(1))
    self.assertEqual(0, non_expired_connection.connection.call_count)