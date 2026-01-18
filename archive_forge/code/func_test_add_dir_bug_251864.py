import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_add_dir_bug_251864(self):
    """Added file turning into a dir should be detected on add dir

        Similar to bug 205636 but with automatic adding of directory contents.
        """
    tree = self.make_branch_and_tree('.')
    self.build_tree(['dir'])
    tree.smart_add(['dir'])
    os.remove('dir')
    self.build_tree(['dir/', 'dir/file'])
    tree.smart_add(['dir'])
    tree.commit('Add dir contents')
    self.addCleanup(tree.lock_read().unlock)
    self.assertEqual([('dir', 'directory'), ('dir/file', 'file')], [(t[0], t[2]) for t in tree.list_files()])
    self.assertFalse(list(tree.iter_changes(tree.basis_tree())))