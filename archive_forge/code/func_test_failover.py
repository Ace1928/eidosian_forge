from unittest import mock
from openstack.block_storage.v3 import service
from openstack.tests.unit import base
@mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=False)
def test_failover(self, mock_supports):
    sot = service.Service(**EXAMPLE)
    res = sot.failover(self.sess)
    self.assertIsNotNone(res)
    url = 'os-services/failover_host'
    body = {'host': 'devstack'}
    self.sess.put.assert_called_with(url, json=body, microversion=self.sess.default_microversion)