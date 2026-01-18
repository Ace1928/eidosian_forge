import os
from breezy import osutils, tests, workingtree
from breezy.tests import features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_add_file_under_symlink(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/link@', 'dir'), ('tree/dir/',), ('tree/dir/file', b'content')])
    if tree.has_versioned_directories():
        self.assertEqual(tree.smart_add(['tree/link/file']), (['dir', 'dir/file'], {}))
    else:
        self.assertEqual(tree.smart_add(['tree/link/file']), (['dir/file'], {}))
    self.assertTrue(tree.is_versioned('dir/file'))
    self.assertTrue(tree.is_versioned('dir'))
    self.assertFalse(tree.is_versioned('link'))
    self.assertFalse(tree.is_versioned('link/file'))