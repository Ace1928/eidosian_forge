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
def restoreDefaults():
    mutabletree.MutableTree.hooks['start_commit'] = []