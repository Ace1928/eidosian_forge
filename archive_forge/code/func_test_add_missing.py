import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_missing(self):
    """brz add foo where foo is missing should error."""
    self.make_branch_and_tree('.')
    self.run_bzr('add missing-file', retcode=3)