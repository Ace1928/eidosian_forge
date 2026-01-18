from openstack.network.v2 import network as _network
from openstack.network.v2 import port as _port
from openstack.network.v2 import tap_flow as _tap_flow
from openstack.network.v2 import tap_service as _tap_service
from openstack.tests.functional import base
def test_list_tap_services(self):
    tap_service_ids = [ts.id for ts in self.user_cloud.network.tap_services()]
    self.assertIn(self.TAP_SERVICE.id, tap_service_ids)