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
def checkEmptyReconcile(self, **kwargs):
    """Check a reconcile on an empty repository."""
    self.make_repository('empty')
    d = BzrDir.open(self.get_url('empty'))
    result = d.find_repository().reconcile(**kwargs)
    self.assertEqual(0, result.inconsistent_parents)
    self.assertEqual(0, result.garbage_inventories)
    self.checkNoBackupInventory(d)