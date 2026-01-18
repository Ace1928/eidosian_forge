import os
from ..workspace import Workspace, WorkspaceDirty, check_clean_tree
from . import TestCaseWithTransport, features, multiply_scenarios
from .scenarios import load_tests_apply_scenarios
def test_dirty(self):
    tree = self.make_branch_and_tree('.', format=self._format)
    self.build_tree(['subpath'])
    self.assertRaises(WorkspaceDirty, Workspace(tree, use_inotify=self._use_inotify).__enter__)