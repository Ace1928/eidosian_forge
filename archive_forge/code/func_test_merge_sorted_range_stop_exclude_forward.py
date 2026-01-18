from breezy import errors, revision, tests
from breezy.tests import per_branch
def test_merge_sorted_range_stop_exclude_forward(self):
    self.assertIterRevids(['2', '1.1.1', '3'], stop_revision_id='1', direction='forward')