import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_add_subdir_file_bug_205636(self):
    """Added file turning into a dir should be detected on add dir/file"""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['dir'])
    tree.smart_add(['dir'])
    os.remove('dir')
    self.build_tree(['dir/', 'dir/file'])
    tree.smart_add(['dir/file'])
    tree.commit('Add file in dir')
    self.addCleanup(tree.lock_read().unlock)
    self.assertEqual([('dir', 'directory'), ('dir/file', 'file')], [(t[0], t[2]) for t in tree.list_files()])
    self.assertFalse(list(tree.iter_changes(tree.basis_tree())))