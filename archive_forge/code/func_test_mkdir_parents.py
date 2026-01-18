import os
from breezy import tests
def test_mkdir_parents(self):
    tree = self.make_branch_and_tree('.')
    self.run_bzr(['mkdir', '-p', 'somedir/foo'])
    self.assertEqual(tree.kind('somedir/foo'), 'directory')