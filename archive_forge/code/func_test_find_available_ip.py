from openstack.network.v2 import floating_ip
from openstack.network.v2 import network
from openstack.network.v2 import port
from openstack.network.v2 import router
from openstack.network.v2 import subnet
from openstack.tests.functional import base
def test_find_available_ip(self):
    sot = self.user_cloud.network.find_available_ip()
    self.assertIsNotNone(sot.id)
    self.assertIsNone(sot.port_id)
    self.assertIsNone(sot.port_details)