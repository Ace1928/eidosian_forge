from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def assertParentsMatch(self, expected_parents_for_versions, repo, when_description):
    for expected_parents, version in expected_parents_for_versions:
        if expected_parents is None:
            self.assertFileVersionAbsent(repo, version)
        else:
            found_parents = self.file_parents(repo, version)
            self.assertEqual(expected_parents, found_parents, '%s reconcile %s has parents %s, should have %s.' % (when_description, version, found_parents, expected_parents))