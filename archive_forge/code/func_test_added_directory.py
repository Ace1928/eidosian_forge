import os
from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def test_added_directory(self):
    """Test --directory option"""
    tree = self.make_branch_and_tree('a')
    self.build_tree(['a/README'])
    tree.add('README')
    out, err = self.run_bzr(['added', '--directory=a'])
    self.assertEqual('README\n', out)