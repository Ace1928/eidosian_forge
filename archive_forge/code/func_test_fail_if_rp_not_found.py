import uuid
from osc_placement.tests.functional import base
def test_fail_if_rp_not_found(self):
    self.assertCommandFailed('No resource provider', self.resource_provider_aggregate_list, 'fake-uuid')