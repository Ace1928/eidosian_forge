import os
from breezy import trace
from breezy.rename_map import RenameMap
from breezy.tests import TestCaseWithTransport
def test_guess_rename_handles_new_directories(self):
    """When a file was moved into a new directory."""
    tree = self.make_branch_and_tree('.')
    tree.lock_write()
    self.build_tree(['file'])
    tree.add('file', ids=b'file-id')
    tree.commit('Added file')
    os.mkdir('folder')
    os.rename('file', 'folder/file2')
    notes = self.captureNotes(RenameMap.guess_renames, tree.basis_tree(), tree)[0]
    self.assertEqual('file => folder/file2', ''.join(notes))
    tree.unlock()