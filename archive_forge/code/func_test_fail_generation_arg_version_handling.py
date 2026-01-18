import uuid
from osc_placement.tests.functional import base
def test_fail_generation_arg_version_handling(self):
    rp = self.resource_provider_create()
    agg = str(uuid.uuid4())
    self.assertCommandFailed('A generation must be specified.', self.resource_provider_aggregate_set, rp['uuid'], agg)