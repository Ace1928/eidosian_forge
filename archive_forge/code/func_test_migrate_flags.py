from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import volume
from openstack.tests.unit import base
def test_migrate_flags(self):
    sot = volume.Volume(**VOLUME)
    self.assertIsNone(sot.migrate(self.sess, host='1', force_host_copy=True, lock_volume=True))
    url = 'volumes/%s/action' % FAKE_ID
    body = {'os-migrate_volume': {'host': '1', 'force_host_copy': True, 'lock_volume': True}}
    self.sess.post.assert_called_with(url, json=body, microversion=sot._max_microversion)