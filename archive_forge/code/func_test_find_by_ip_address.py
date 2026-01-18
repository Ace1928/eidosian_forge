from openstack.network.v2 import floating_ip
from openstack.network.v2 import network
from openstack.network.v2 import port
from openstack.network.v2 import router
from openstack.network.v2 import subnet
from openstack.tests.functional import base
def test_find_by_ip_address(self):
    sot = self.user_cloud.network.find_ip(self.FIP.floating_ip_address)
    self.assertEqual(self.FIP.floating_ip_address, sot.floating_ip_address)
    self.assertEqual(self.FIP.floating_ip_address, sot.name)