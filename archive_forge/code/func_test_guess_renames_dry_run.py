import os
from breezy import trace
from breezy.rename_map import RenameMap
from breezy.tests import TestCaseWithTransport
def test_guess_renames_dry_run(self):
    tree = self.make_branch_and_tree('tree')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    self.build_tree(['tree/file'])
    tree.add('file', ids=b'file-id')
    tree.commit('Added file')
    os.rename('tree/file', 'tree/file2')
    RenameMap.guess_renames(tree.basis_tree(), tree, dry_run=True)
    self.assertEqual('file', tree.id2path(b'file-id'))