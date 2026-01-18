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
def test_commit_deleted_subtree_and_files_updates_workingtree(self):
    """The working trees inventory may be adjusted by commit."""
    wt = self.make_branch_and_tree('.')
    wt.lock_write()
    self.build_tree(['a', 'b/', 'b/c', 'd'])
    wt.add(['a', 'b', 'b/c', 'd'])
    this_dir = wt.controldir.root_transport
    this_dir.delete_tree('b')
    this_dir.delete('d')
    wt.commit('commit stuff')
    self.assertTrue(wt.has_filename('a'))
    self.assertFalse(wt.has_filename('b'))
    self.assertFalse(wt.has_filename('b/c'))
    self.assertFalse(wt.has_filename('d'))
    wt.unlock()
    wt = wt.controldir.open_workingtree()
    with wt.lock_read():
        self.assertTrue(wt.has_filename('a'))
        self.assertFalse(wt.has_filename('b'))
        self.assertFalse(wt.has_filename('b/c'))
        self.assertFalse(wt.has_filename('d'))