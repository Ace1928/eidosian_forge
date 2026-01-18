import uuid
from keystoneclient.tests.unit.v3 import test_endpoint_filter
from keystoneclient.tests.unit.v3 import utils
def test_create_policy_association_for_region_and_service_via_obj(self):
    self._crud_policy_association_for_region_and_service_via_obj('PUT', self.manager.create_policy_association_for_region_and_service)