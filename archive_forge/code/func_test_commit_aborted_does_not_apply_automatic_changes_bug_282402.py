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
def test_commit_aborted_does_not_apply_automatic_changes_bug_282402(self):
    wt = self.make_branch_and_tree('.')
    wt.add(['a'], ['file'])
    self.assertTrue(wt.is_versioned('a'))
    if wt.supports_setting_file_ids():
        a_id = wt.path2id('a')
        self.assertEqual('a', wt.id2path(a_id))

    def fail_message(obj):
        raise errors.CommandError('empty commit message')
    self.assertRaises(errors.CommandError, wt.commit, message_callback=fail_message)
    self.assertTrue(wt.is_versioned('a'))
    if wt.supports_setting_file_ids():
        self.assertEqual('a', wt.id2path(a_id))