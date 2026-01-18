import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_new_dirstate_on_new_lock(self):
    known_dirstates = set()

    def lock_and_compare_all_current_dirstate(tree, lock_method):
        getattr(tree, lock_method)()
        state = tree.current_dirstate()
        self.assertFalse(state in known_dirstates)
        known_dirstates.add(state)
        tree.unlock()
    tree = self.make_workingtree()
    lock_and_compare_all_current_dirstate(tree, 'lock_read')
    lock_and_compare_all_current_dirstate(tree, 'lock_read')
    lock_and_compare_all_current_dirstate(tree, 'lock_tree_write')
    lock_and_compare_all_current_dirstate(tree, 'lock_tree_write')
    lock_and_compare_all_current_dirstate(tree, 'lock_write')
    lock_and_compare_all_current_dirstate(tree, 'lock_write')