from breezy import errors, revision, tests
from breezy.tests import per_branch
def test_merge_sorted_range_stop_with_merges_can_show_non_parents(self):
    self.assertIterRevids(['3', '1.1.1', '2'], stop_revision_id='2', stop_rule='with-merges')