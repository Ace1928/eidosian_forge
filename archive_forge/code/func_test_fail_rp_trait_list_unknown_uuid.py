import uuid
from osc_placement.tests.functional import base
def test_fail_rp_trait_list_unknown_uuid(self):
    self.assertCommandFailed('No resource provider', self.resource_provider_trait_list, 123)