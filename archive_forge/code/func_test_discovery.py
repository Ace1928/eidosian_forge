import ddt
from keystoneauth1 import exceptions
from openstack.tests.unit import base
def test_discovery(self):
    self._register_uris()
    rs = self.cloud.placement.get('/allocation_candidates')
    self._validate_resp(rs, 200)