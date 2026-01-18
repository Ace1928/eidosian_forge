from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_check_behaviour(self):
    """Populate a repository and check it, and verify the output."""
    repo, scenario = self.prepare_test_repository()
    check_result = repo.check()
    check_result.report_results(verbose=True)
    log = self.get_log()
    for pattern in scenario.check_regexes(repo):
        self.assertContainsRe(log, pattern)