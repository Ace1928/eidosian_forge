import collections
import copy
import uuid
from osc_placement.tests.functional import base
def test_fail_aggregate_arg_version_handling(self):
    agg = str(uuid.uuid4())
    self.assertCommandFailed('Operation or argument is not supported with version 1.0; requires at least version 1.3', self.resource_inventory_set, agg, 'MEMORY_MB=16', aggregate=True)