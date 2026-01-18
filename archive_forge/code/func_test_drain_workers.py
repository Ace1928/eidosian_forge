from unittest import mock
from glance.api import common
from glance.api.v2 import cached_images
import glance.async_
from glance.common import exception
from glance.common import wsgi_app
from glance import sqlite_migration
from glance.tests import utils as test_utils
@mock.patch('glance.api.v2.cached_images.WORKER')
@mock.patch('glance.async_._THREADPOOL_MODEL', new=None)
def test_drain_workers(self, mock_cache_worker):
    glance.async_.set_threadpool_model('native')
    model = common.get_thread_pool('tasks_pool')
    with mock.patch.object(model.pool, 'shutdown') as mock_shutdown:
        wsgi_app.drain_workers()
        mock_shutdown.assert_called_once_with()
        mock_cache_worker.terminate.assert_called_once_with()