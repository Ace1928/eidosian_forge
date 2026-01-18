import os
from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def test_deleted_directory(self):
    """Test --directory option"""
    tree = self.make_branch_and_tree('a')
    self.build_tree(['a/README'])
    tree.add('README')
    tree.commit('r1')
    tree.remove('README')
    out, err = self.run_bzr(['deleted', '--directory=a'])
    self.assertEqual('README\n', out)