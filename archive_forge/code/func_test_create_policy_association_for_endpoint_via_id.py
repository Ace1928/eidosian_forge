import uuid
from keystoneclient.tests.unit.v3 import test_endpoint_filter
from keystoneclient.tests.unit.v3 import utils
def test_create_policy_association_for_endpoint_via_id(self):
    self._crud_policy_association_for_endpoint_via_id('PUT', self.manager.create_policy_association_for_endpoint)