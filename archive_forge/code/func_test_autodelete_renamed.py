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
def test_autodelete_renamed(self):
    tree_a = self.make_branch_and_tree('a')
    self.build_tree(['a/dir/', 'a/dir/f1', 'a/dir/f2'])
    tree_a.add(['dir', 'dir/f1', 'dir/f2'])
    rev_id1 = tree_a.commit('init')
    tree_a.rename_one('dir/f1', 'dir/a')
    tree_a.rename_one('dir/f2', 'dir/z')
    osutils.rmtree('a/dir')
    tree_a.commit('autoremoved')
    with tree_a.lock_read():
        paths = [(path, ie.file_id) for path, ie in tree_a.iter_entries_by_dir()]
    if tree_a.supports_file_ids:
        self.assertEqual([('', tree_a.path2id(''))], paths)