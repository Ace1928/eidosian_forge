from unittest import mock
from os_brick import caches
from os_brick import exception
from os_brick.tests import base
@mock.patch('os_brick.executor.Executor._execute')
def test_init_invalid_device_path(self, moc_exec):
    conn_info_invalid = {'data': {}}
    self.assertRaises(exception.VolumeLocalCacheNotSupported, caches.CacheManager, root_helper=None, connection_info=conn_info_invalid)