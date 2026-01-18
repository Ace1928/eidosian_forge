import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.tests.unit import base
def test_create_router_wrong_availability_zone_hints_type(self):
    azh_opts = 'invalid'
    with testtools.ExpectedException(exceptions.SDKException, "Parameter 'availability_zone_hints' must be a list"):
        self.cloud.create_router(name=self.router_name, admin_state_up=True, availability_zone_hints=azh_opts)