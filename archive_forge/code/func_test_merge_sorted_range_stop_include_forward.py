from breezy import errors, revision, tests
from breezy.tests import per_branch
def test_merge_sorted_range_stop_include_forward(self):
    self.assertIterRevids(['2', '1.1.1', '3'], stop_revision_id='2', stop_rule='include', direction='forward')