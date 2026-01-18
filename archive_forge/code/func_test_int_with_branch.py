import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_int_with_branch(self):
    revinfo = self.get_in_history('2:tree2')
    self.assertNotEqual(self.tree.branch.base, revinfo.branch.base)
    self.assertEqual(self.tree2.branch.base, revinfo.branch.base)
    self.assertEqual(2, revinfo.revno)
    self.assertEqual(b'alt_r2', revinfo.rev_id)