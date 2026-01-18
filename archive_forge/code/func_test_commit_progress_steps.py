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
def test_commit_progress_steps(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b', 'c'])
    tree.add(['a', 'b', 'c'])
    tree.commit('first post')
    with open('b', 'w') as f:
        f.write('new content')
    factory = ProgressRecordingUIFactory()
    ui.ui_factory = factory
    tree.commit('second post', specific_files=['b'])
    self.assertEqual([('update', 1, 5, 'Collecting changes [0] - Stage'), ('update', 1, 5, 'Collecting changes [1] - Stage'), ('update', 2, 5, 'Saving data locally - Stage'), ('update', 3, 5, 'Running pre_commit hooks - Stage'), ('update', 4, 5, 'Updating the working tree - Stage'), ('update', 5, 5, 'Running post_commit hooks - Stage')], factory._calls)