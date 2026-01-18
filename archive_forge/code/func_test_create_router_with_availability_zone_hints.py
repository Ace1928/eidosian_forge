import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.tests.unit import base
def test_create_router_with_availability_zone_hints(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': self.enabled_neutron_extensions}), dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'routers']), json={'router': self.mock_router_rep}, validate=dict(json={'router': {'name': self.router_name, 'admin_state_up': True, 'availability_zone_hints': ['nova']}}))])
    self.cloud.create_router(name=self.router_name, admin_state_up=True, availability_zone_hints=['nova'])
    self.assert_calls()