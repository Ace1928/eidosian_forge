from breezy import errors, revision, tests
from breezy.tests import per_branch
def test_merge_sorted_range_start_only_forward(self):
    self.assertIterRevids(['1', '1.1.1'], start_revision_id='1.1.1', direction='forward')