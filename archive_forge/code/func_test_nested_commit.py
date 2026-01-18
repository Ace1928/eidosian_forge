import os
from breezy import branch, conflicts, controldir, errors, mutabletree, osutils
from breezy import revision as _mod_revision
from breezy import tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.commit import CannotCommitSelectedFileMerge, PointlessCommit
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.tests.testui import ProgressRecordingUIFactory
def test_nested_commit(self):
    """Commit in multiply-nested trees"""
    tree = self.make_branch_and_tree('.')
    if not tree.supports_tree_reference():
        return
    subtree = self.make_branch_and_tree('subtree')
    subsubtree = self.make_branch_and_tree('subtree/subtree')
    subsub_revid = subsubtree.commit('subsubtree')
    subtree.commit('subtree')
    subtree.add(['subtree'])
    tree.add(['subtree'])
    rev_id = tree.commit('added reference', allow_pointless=False)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual(subsub_revid, subsubtree.last_revision())
    sub_basis = subtree.basis_tree()
    sub_basis.lock_read()
    self.addCleanup(sub_basis.unlock)
    self.assertEqual(subsubtree.last_revision(), sub_basis.get_reference_revision('subtree'))
    self.assertNotEqual(None, subtree.last_revision())
    basis = tree.basis_tree()
    basis.lock_read()
    self.addCleanup(basis.unlock)
    self.assertEqual(subtree.last_revision(), basis.get_reference_revision('subtree'))
    self.assertNotEqual(None, rev_id)