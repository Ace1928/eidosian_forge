import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.tests.unit import base
def test_create_router_with_enable_snat_False(self):
    """Send enable_snat when it is False."""
    self.register_uris([dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'routers']), json={'router': self.mock_router_rep}, validate=dict(json={'router': {'name': self.router_name, 'external_gateway_info': {'enable_snat': False}, 'admin_state_up': True}}))])
    self.cloud.create_router(name=self.router_name, admin_state_up=True, enable_snat=False)
    self.assert_calls()