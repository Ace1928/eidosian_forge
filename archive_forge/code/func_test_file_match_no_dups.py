import os
from breezy import trace
from breezy.rename_map import RenameMap
from breezy.tests import TestCaseWithTransport
def test_file_match_no_dups(self):
    tree = self.make_branch_and_tree('tree')
    rn = RenameMap(tree)
    rn.add_edge_hashes(self.a_lines, 'aid')
    self.build_tree_contents([('tree/a', b''.join(self.a_lines))])
    self.build_tree_contents([('tree/b', b''.join(self.b_lines))])
    self.build_tree_contents([('tree/c', b''.join(self.b_lines))])
    self.assertEqual({'a': 'aid'}, rn.file_match(['a', 'b', 'c']))