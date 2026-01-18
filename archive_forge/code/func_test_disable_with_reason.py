from unittest import mock
from openstack.compute.v2 import service
from openstack import exceptions
from openstack.tests.unit import base
def test_disable_with_reason(self):
    sot = service.Service(**EXAMPLE)
    reason = 'fencing'
    res = sot.disable(self.sess, 'host1', 'nova-compute', reason=reason)
    self.assertIsNotNone(res)
    url = 'os-services/disable-log-reason'
    body = {'binary': 'nova-compute', 'host': 'host1', 'disabled_reason': reason}
    self.sess.put.assert_called_with(url, json=body, microversion=self.sess.default_microversion)