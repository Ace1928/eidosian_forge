from breezy import errors, revision, tests
from breezy.tests import per_branch
def test_merge_sorted_forward(self):
    self.assertIterRevids(['1', '2', '1.1.1', '3'], direction='forward')