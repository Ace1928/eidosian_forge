from unittest import mock
from openstack.compute.v2 import flavor
from openstack.compute.v2 import server
from openstack.image.v2 import image
from openstack.tests.unit import base
def test_live_migrate_30_force(self):
    sot = server.Server(**EXAMPLE)

    class FakeEndpointData:
        min_microversion = '2.1'
        max_microversion = '2.30'
    self.sess.get_endpoint_data.return_value = FakeEndpointData()
    self.sess.default_microversion = None
    res = sot.live_migrate(self.sess, host='HOST2', force=True, block_migration=None)
    self.assertIsNone(res)
    url = 'servers/IDENTIFIER/action'
    body = {'os-migrateLive': {'block_migration': 'auto', 'host': 'HOST2', 'force': True}}
    headers = {'Accept': ''}
    self.sess.post.assert_called_with(url, json=body, headers=headers, microversion='2.30')