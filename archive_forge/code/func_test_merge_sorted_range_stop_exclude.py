from breezy import errors, revision, tests
from breezy.tests import per_branch
def test_merge_sorted_range_stop_exclude(self):
    self.assertIterRevids(['3', '1.1.1', '2'], stop_revision_id='1')