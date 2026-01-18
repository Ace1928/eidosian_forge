import os
from breezy.tests import TestCaseWithTransport
def test_ancestry_with_checkout(self):
    """Tests 'ancestry' command with a location that is a
        checkout of a repository branch."""
    a_tree = self._build_branches()[0]
    self.make_repository('repo', shared=True)
    repo_branch = a_tree.controldir.sprout('repo/A').open_branch()
    repo_branch.create_checkout('A-checkout')
    self._check_ancestry('A-checkout')