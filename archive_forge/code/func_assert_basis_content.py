import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def assert_basis_content(self, expected_content, branch, path):
    basis = branch.basis_tree()
    with basis.lock_read():
        self.assertEqual(expected_content, basis.get_file_text(path))