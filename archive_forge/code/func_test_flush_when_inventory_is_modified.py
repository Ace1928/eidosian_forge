import sys
from breezy import errors
from breezy.tests import TestSkipped
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_flush_when_inventory_is_modified(self):
    if sys.platform == 'win32':
        raise TestSkipped("don't use oslocks on win32 in unix manner")
    self.thisFailsStrictLockCheck()
    tree = self.make_branch_and_tree('tree')
    with tree.lock_write():
        if tree.supports_file_ids:
            old_root = tree.path2id('')
        tree.add('')
        reference_tree = tree.controldir.open_workingtree()
        if tree.supports_file_ids:
            self.assertEqual(old_root, reference_tree.path2id(''))
        tree.flush()
        reference_tree = tree.controldir.open_workingtree()
        self.assertTrue(reference_tree.is_versioned(''))
        if reference_tree.supports_file_ids:
            self.assertIsNot(None, reference_tree.path2id(''))