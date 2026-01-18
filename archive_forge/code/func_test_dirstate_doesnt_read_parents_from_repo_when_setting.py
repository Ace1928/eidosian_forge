import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_dirstate_doesnt_read_parents_from_repo_when_setting(self):
    """Setting parent trees on a dirstate working tree takes
        the trees it's given and doesn't need to read them from the
        repository.
        """
    tree = self.make_workingtree()
    subtree = self.make_branch_and_tree('subdir')
    rev1 = subtree.commit('commit in subdir')
    rev1_tree = subtree.basis_tree()
    rev1_tree.lock_read()
    self.addCleanup(rev1_tree.unlock)
    tree.branch.pull(subtree.branch)
    repo = tree.branch.repository
    self.overrideAttr(repo, 'get_revision', self.fail)
    self.overrideAttr(repo, 'get_inventory', self.fail)
    self.overrideAttr(repo, '_get_inventory_xml', self.fail)
    tree.set_parent_trees([(rev1, rev1_tree)])