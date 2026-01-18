from openstack.network.v2 import network as _network
from openstack.network.v2 import port as _port
from openstack.network.v2 import tap_flow as _tap_flow
from openstack.network.v2 import tap_service as _tap_service
from openstack.tests.functional import base
def test_find_tap_service(self):
    sot = self.user_cloud.network.find_tap_service(self.TAP_SERVICE.name)
    self.assertEqual(self.SERVICE_PORT_ID, sot.port_id)
    self.assertEqual(self.TAP_S_NAME, sot.name)