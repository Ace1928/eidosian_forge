from unittest import mock
from os_brick import caches
from os_brick import exception
from os_brick.tests import base
@mock.patch('os_brick.caches.CacheManager._get_engine')
def test_init_cacheable(self, moc_get_engine):
    moc_get_engine.return_value = None
    conn_info_cacheable = {'data': {'device_path': '/dev/sdd', 'cacheable': True}}
    conn_info_non_cacheable = {'data': {'device_path': '/dev/sdd'}}
    mgr_cacheable = caches.CacheManager(root_helper=None, connection_info=conn_info_cacheable)
    mgr_non_cacheable = caches.CacheManager(root_helper=None, connection_info=conn_info_non_cacheable)
    self.assertTrue(mgr_cacheable.cacheable)
    self.assertFalse(mgr_non_cacheable.cacheable)