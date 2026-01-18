from openstack.network.v2 import flavor
from openstack.tests.functional import base
def test_associate_disassociate_flavor_with_service_profile(self):
    if not self.operator_cloud:
        self.skipTest('Operator cloud required for this test')
    response = self.operator_cloud.network.associate_flavor_with_service_profile(self.ID, self.service_profiles.id)
    self.assertIsNotNone(response)
    response = self.operator_cloud.network.disassociate_flavor_from_service_profile(self.ID, self.service_profiles.id)
    self.assertIsNone(response)