from breezy import branch, tests
from breezy.pyutils import calc_parent_name, get_named_object
def test_dotted_attr(self):
    self.assertIs(branch.Branch.hooks, get_named_object('breezy.branch', 'Branch.hooks'))