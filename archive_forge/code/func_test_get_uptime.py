import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import hypervisor
from openstack import exceptions
from openstack.tests.unit import base
@mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=False)
def test_get_uptime(self, mv_mock):
    sot = hypervisor.Hypervisor(**copy.deepcopy(EXAMPLE))
    rsp = {'hypervisor': {'hypervisor_hostname': 'fake-mini', 'id': sot.id, 'state': 'up', 'status': 'enabled', 'uptime': '08:32:11 up 93 days, 18:25, 12 users'}}
    resp = mock.Mock()
    resp.body = copy.deepcopy(rsp)
    resp.json = mock.Mock(return_value=resp.body)
    resp.headers = {}
    resp.status_code = 200
    self.sess.get = mock.Mock(return_value=resp)
    hyp = sot.get_uptime(self.sess)
    self.sess.get.assert_called_with('os-hypervisors/{id}/uptime'.format(id=sot.id), microversion=self.sess.default_microversion)
    self.assertEqual(rsp['hypervisor']['uptime'], hyp.uptime)
    self.assertEqual(rsp['hypervisor']['status'], sot.status)