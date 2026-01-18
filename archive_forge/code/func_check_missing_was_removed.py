import breezy
from breezy import errors
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.inventory import Inventory
from breezy.bzr.tests.per_repository_vf import (
from breezy.bzr.tests.per_repository_vf.helpers import \
from breezy.reconcile import Reconciler, reconcile
from breezy.revision import Revision
from breezy.tests import TestSkipped
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
from breezy.uncommit import uncommit
def check_missing_was_removed(self, repo):
    if repo._reconcile_backsup_inventory:
        backed_up = False
        for path in repo.control_transport.list_dir('.'):
            if 'inventory.backup' in path:
                backed_up = True
        self.assertTrue(backed_up)
    self.assertRaises(errors.NoSuchRevision, repo.get_inventory, 'missing')