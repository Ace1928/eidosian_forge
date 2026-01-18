from breezy import errors, revision, tests
from breezy.tests import per_branch
def test_merge_sorted_single_stop_exclude(self):
    self.assertIterRevids([], start_revision_id='3', stop_revision_id='3')