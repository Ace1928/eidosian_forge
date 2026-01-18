from unittest import mock
from openstack.block_storage.v3 import service
from openstack.tests.unit import base
def test_disable__with_reason(self):
    sot = service.Service(**EXAMPLE)
    reason = 'fencing'
    res = sot.disable(self.sess, reason=reason)
    self.assertIsNotNone(res)
    url = 'os-services/disable-log-reason'
    body = {'binary': 'cinder-scheduler', 'host': 'devstack', 'disabled_reason': reason}
    self.sess.put.assert_called_with(url, json=body, microversion=self.sess.default_microversion)