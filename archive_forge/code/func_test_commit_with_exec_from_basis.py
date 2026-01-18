import os
from breezy import osutils
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_commit_with_exec_from_basis(self):
    self.wt._is_executable_from_path_and_stat = self.wt._is_executable_from_path_and_stat_from_basis
    rev_id1 = self.wt.commit('one')
    rev_tree1 = self.wt.branch.repository.revision_tree(rev_id1)
    a_executable = rev_tree1.is_executable('a')
    b_executable = rev_tree1.is_executable('b')
    self.assertIsNot(None, a_executable)
    self.assertTrue(a_executable)
    self.assertIsNot(None, b_executable)
    self.assertFalse(b_executable)