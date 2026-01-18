import uuid
from osc_placement.tests.functional import base
def test_list_empty(self):
    self.assertEqual([], self.allocation_candidate_list(resources=['MEMORY_MB=999999999']))