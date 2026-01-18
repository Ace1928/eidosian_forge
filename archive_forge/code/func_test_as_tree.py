import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_as_tree(self):
    tree = self.get_as_tree('branch:tree', self.tree2)
    self.assertEqual(b'r2', tree.get_revision_id())
    self.assertFalse(self.tree2.branch.repository.has_revision(b'r2'))