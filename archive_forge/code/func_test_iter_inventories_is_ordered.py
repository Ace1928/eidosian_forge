from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_iter_inventories_is_ordered(self):
    tree = self.make_branch_and_tree('a')
    first_revision = tree.commit('')
    second_revision = tree.commit('')
    tree.lock_read()
    self.addCleanup(tree.unlock)
    revs = (first_revision, second_revision)
    invs = tree.branch.repository.iter_inventories(revs)
    for rev_id, inv in zip(revs, invs):
        self.assertEqual(rev_id, inv.revision_id)
        self.assertIsInstance(inv, inventory.CommonInventory)