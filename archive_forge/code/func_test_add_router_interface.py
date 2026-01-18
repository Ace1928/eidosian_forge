import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.tests.unit import base
def test_add_router_interface(self):
    self.register_uris([dict(method='PUT', uri=self.get_mock_url('network', 'public', append=['v2.0', 'routers', self.router_id, 'add_router_interface']), json={'port': self.mock_router_interface_rep}, validate=dict(json={'subnet_id': self.subnet_id}))])
    self.cloud.add_router_interface({'id': self.router_id}, subnet_id=self.subnet_id)
    self.assert_calls()