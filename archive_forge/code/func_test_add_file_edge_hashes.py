import os
from breezy import trace
from breezy.rename_map import RenameMap
from breezy.tests import TestCaseWithTransport
def test_add_file_edge_hashes(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/a', b''.join(self.a_lines))])
    tree.add('a', ids=b'a')
    rn = RenameMap(tree)
    rn.add_file_edge_hashes(tree, [b'a'])
    self.assertEqual({b'a'}, rn.edge_hashes[myhash(('a\n', 'b\n'))])
    self.assertEqual({b'a'}, rn.edge_hashes[myhash(('b\n', 'c\n'))])
    self.assertIs(None, rn.edge_hashes.get(myhash(('c\n', 'd\n'))))