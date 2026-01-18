import os
from breezy import trace
from breezy.rename_map import RenameMap
from breezy.tests import TestCaseWithTransport
def test_match_directories(self):
    tree = self.make_branch_and_tree('tree')
    rn = RenameMap(tree)
    required_parents = rn.get_required_parents({'path1': 'a', 'path2/tr': 'b', 'path3/path4/path5': 'c'})
    self.assertEqual({'path2': {'b'}, 'path3/path4': {'c'}, 'path3': set()}, required_parents)