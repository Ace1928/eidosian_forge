import os
from breezy.branch import Branch
from breezy.osutils import pathjoin
from breezy.tests import TestCaseInTempDir, TestCaseWithTransport
from breezy.trace import mutter
from breezy.workingtree import WorkingTree
def test_mkdir_w_nested_trees(self):
    """'brz mkdir' with nested trees"""
    self.make_branch_and_tree('.')
    self.make_branch_and_tree('a')
    self.make_branch_and_tree('a/b')
    self.run_bzr(['mkdir', 'dir', 'a/dir', 'a/b/dir'])
    self.assertTrue(os.path.isdir('dir'))
    self.assertTrue(os.path.isdir('a/dir'))
    self.assertTrue(os.path.isdir('a/b/dir'))
    wt = WorkingTree.open('.')
    wt_a = WorkingTree.open('a')
    wt_b = WorkingTree.open('a/b')
    delta = wt.changes_from(wt.basis_tree())
    self.assertEqual(len(delta.added), 1)
    self.assertEqual(delta.added[0].path[1], 'dir')
    self.assertFalse(delta.modified)
    delta = wt_a.changes_from(wt_a.basis_tree())
    self.assertEqual(len(delta.added), 1)
    self.assertEqual(delta.added[0].path[1], 'dir')
    self.assertFalse(delta.modified)
    delta = wt_b.changes_from(wt_b.basis_tree())
    self.assertEqual(len(delta.added), 1)
    self.assertEqual(delta.added[0].path[1], 'dir')
    self.assertFalse(delta.modified)