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
def test_convenience_reconcile_inventory_without_revision_reconcile(self):
    bzrdir_url = self.get_url('inventory_without_revision')
    bzrdir = BzrDir.open(bzrdir_url)
    repo = bzrdir.open_repository()
    if not repo._reconcile_does_inventory_gc:
        raise TestSkipped('Irrelevant test')
    reconcile(bzrdir)
    repo = bzrdir.open_repository()
    self.check_missing_was_removed(repo)