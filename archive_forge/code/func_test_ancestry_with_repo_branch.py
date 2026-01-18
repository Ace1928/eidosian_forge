import os
from breezy.tests import TestCaseWithTransport
def test_ancestry_with_repo_branch(self):
    """Tests 'ancestry' command with a location that is a
        repository branch."""
    a_tree = self._build_branches()[0]
    self.make_repository('repo', shared=True)
    a_tree.controldir.sprout('repo/A')
    self._check_ancestry('repo/A')