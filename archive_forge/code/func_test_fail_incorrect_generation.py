import uuid
from osc_placement.tests.functional import base
def test_fail_incorrect_generation(self):
    rp = self.resource_provider_create()
    agg = str(uuid.uuid4())
    self.assertCommandFailed('Please update the generation and try again.', self.resource_provider_aggregate_set, rp['uuid'], agg, generation=99999)