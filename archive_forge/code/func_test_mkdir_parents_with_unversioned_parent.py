import os
from breezy import tests
def test_mkdir_parents_with_unversioned_parent(self):
    tree = self.make_branch_and_tree('.')
    os.mkdir('somedir')
    self.run_bzr(['mkdir', '-p', 'somedir/foo'])
    self.assertEqual(tree.kind('somedir/foo'), 'directory')