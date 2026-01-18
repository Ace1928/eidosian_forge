import os
from breezy import tests
def test_mkdir_parents_existing_versioned_dir(self):
    tree = self.make_branch_and_tree('.')
    tree.mkdir('somedir')
    self.assertEqual(tree.kind('somedir'), 'directory')
    self.run_bzr(['mkdir', '-p', 'somedir'])