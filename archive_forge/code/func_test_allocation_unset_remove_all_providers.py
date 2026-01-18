import uuid
from osc_placement.tests.functional import base
def test_allocation_unset_remove_all_providers(self):
    """Tests removing all allocations by omitting the --provider option."""
    updated_allocs = self.resource_allocation_unset(self.consumer_uuid1, use_json=False)
    self.assertEqual('', updated_allocs.strip())