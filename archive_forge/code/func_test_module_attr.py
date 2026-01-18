from breezy import branch, tests
from breezy.pyutils import calc_parent_name, get_named_object
def test_module_attr(self):
    self.assertIs(branch.Branch, get_named_object('breezy.branch', 'Branch'))