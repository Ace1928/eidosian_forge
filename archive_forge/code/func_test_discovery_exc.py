import ddt
from keystoneauth1 import exceptions
from openstack.tests.unit import base
def test_discovery_exc(self):
    self._register_uris(status_code=500)
    ex = self.assertRaises(exceptions.InternalServerError, self.cloud.placement.get, '/allocation_candidates', raise_exc=True)
    self._validate_resp(ex.response, 500)