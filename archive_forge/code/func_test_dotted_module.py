from breezy import branch, tests
from breezy.pyutils import calc_parent_name, get_named_object
def test_dotted_module(self):
    self.assertIs(branch, get_named_object('breezy.branch'))