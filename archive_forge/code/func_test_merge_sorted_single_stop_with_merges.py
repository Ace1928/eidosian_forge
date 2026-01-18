from breezy import errors, revision, tests
from breezy.tests import per_branch
def test_merge_sorted_single_stop_with_merges(self):
    self.assertIterRevids(['3', '1.1.1'], start_revision_id='3', stop_revision_id='3', stop_rule='with-merges')