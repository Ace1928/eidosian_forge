from openstack.network.v2 import network as _network
from openstack.network.v2 import port as _port
from openstack.network.v2 import tap_flow as _tap_flow
from openstack.network.v2 import tap_service as _tap_service
from openstack.tests.functional import base
def test_get_tap_flow(self):
    sot = self.user_cloud.network.get_tap_flow(self.TAP_FLOW.id)
    self.assertEqual(self.FLOW_PORT_ID, sot.source_port)
    self.assertEqual(self.TAP_F_NAME, sot.name)
    self.assertEqual(self.TAP_SERVICE.id, sot.tap_service_id)
    self.assertEqual('BOTH', sot.direction)