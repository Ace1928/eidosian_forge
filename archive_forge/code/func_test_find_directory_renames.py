import os
from breezy import trace
from breezy.rename_map import RenameMap
from breezy.tests import TestCaseWithTransport
def test_find_directory_renames(self):
    tree = self.make_branch_and_tree('tree')
    rn = RenameMap(tree)
    matches = {'path1': 'a', 'path3/path4/path5': 'c'}
    required_parents = {'path2': {'b'}, 'path3/path4': {'c'}, 'path3': set()}
    missing_parents = {'path2-id': {'b'}, 'path4-id': {'c'}, 'path3-id': {'path4-id'}}
    matches = rn.match_parents(required_parents, missing_parents)
    self.assertEqual({'path3/path4': 'path4-id', 'path2': 'path2-id'}, matches)