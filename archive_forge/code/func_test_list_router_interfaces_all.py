import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.tests.unit import base
def test_list_router_interfaces_all(self):
    self._test_list_router_interfaces(self.router, interface_type=None)