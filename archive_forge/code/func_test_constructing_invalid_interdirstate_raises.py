import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_constructing_invalid_interdirstate_raises(self):
    tree = self.make_workingtree()
    rev_id = tree.commit('first post')
    tree.commit('second post')
    rev_tree = tree.branch.repository.revision_tree(rev_id)
    self.assertRaises(Exception, workingtree_4.InterDirStateTree, rev_tree, tree)
    self.assertRaises(Exception, workingtree_4.InterDirStateTree, tree, rev_tree)