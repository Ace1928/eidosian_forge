import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_dirstate_doesnt_cache_non_parent_trees(self):
    """Getting parent trees from a dirstate tree does not read from the
        repos inventory store. This is an important part of the dirstate
        performance optimisation work.
        """
    tree = self.make_workingtree()
    subtree = self.make_branch_and_tree('subdir')
    rev1 = subtree.commit('commit in subdir')
    tree.branch.pull(subtree.branch)
    self.assertRaises(errors.NoSuchRevision, tree.revision_tree, rev1)