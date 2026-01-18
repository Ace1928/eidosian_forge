from openstack.network.v2 import network as _network
from openstack.network.v2 import port as _port
from openstack.network.v2 import tap_flow as _tap_flow
from openstack.network.v2 import tap_service as _tap_service
from openstack.tests.functional import base
def test_list_tap_flows(self):
    tap_flow_ids = [tf.id for tf in self.user_cloud.network.tap_flows()]
    self.assertIn(self.TAP_FLOW.id, tap_flow_ids)