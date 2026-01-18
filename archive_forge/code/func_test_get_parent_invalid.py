from breezy import branch, errors
from breezy.tests import per_branch, test_server
def test_get_parent_invalid(self):
    branch_b = self.get_branch_with_invalid_parent()
    self.assertRaises(errors.InaccessibleParent, branch_b.get_parent)