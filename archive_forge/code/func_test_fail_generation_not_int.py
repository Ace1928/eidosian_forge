import uuid
from osc_placement.tests.functional import base
def test_fail_generation_not_int(self):
    rp = self.resource_provider_create()
    agg = str(uuid.uuid4())
    self.assertCommandFailed('invalid int value', self.resource_provider_aggregate_set, rp['uuid'], agg, generation='barney')