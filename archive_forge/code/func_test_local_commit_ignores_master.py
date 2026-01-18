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
def test_local_commit_ignores_master(self):
    master = self.make_branch('master')
    tree = self.make_branch_and_tree('tree')
    try:
        tree.branch.bind(master)
    except branch.BindingUnsupported:
        return
    master.controldir.transport.put_bytes('branch-format', b'garbage')
    del master
    self.assertRaises(errors.UnknownFormatError, controldir.ControlDir.open, 'master')
    tree.commit('foo', local=True)