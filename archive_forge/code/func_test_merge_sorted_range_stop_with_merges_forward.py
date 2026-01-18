from breezy import errors, revision, tests
from breezy.tests import per_branch
def test_merge_sorted_range_stop_with_merges_forward(self):
    self.assertIterRevids(['1.1.1', '3'], stop_revision_id='3', stop_rule='with-merges', direction='forward')