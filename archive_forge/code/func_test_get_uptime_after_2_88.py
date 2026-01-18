import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import hypervisor
from openstack import exceptions
from openstack.tests.unit import base
@mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=True)
def test_get_uptime_after_2_88(self, mv_mock):
    sot = hypervisor.Hypervisor(**copy.deepcopy(EXAMPLE))
    self.assertRaises(exceptions.SDKException, sot.get_uptime, self.sess)