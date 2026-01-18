import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_no_dirstate_outside_lock(self):
    """Getting a dirstate object fails if there is no lock."""

    def lock_and_call_current_dirstate(tree, lock_method):
        getattr(tree, lock_method)()
        tree.current_dirstate()
        tree.unlock()
    tree = self.make_workingtree()
    self.assertRaises(errors.ObjectNotLocked, tree.current_dirstate)
    lock_and_call_current_dirstate(tree, 'lock_read')
    self.assertRaises(errors.ObjectNotLocked, tree.current_dirstate)
    lock_and_call_current_dirstate(tree, 'lock_write')
    self.assertRaises(errors.ObjectNotLocked, tree.current_dirstate)
    lock_and_call_current_dirstate(tree, 'lock_tree_write')
    self.assertRaises(errors.ObjectNotLocked, tree.current_dirstate)