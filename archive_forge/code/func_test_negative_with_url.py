import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_negative_with_url(self):
    url = self.get_url() + '/tree2'
    revinfo = self.get_in_history('revno:-1:{}'.format(url))
    self.assertNotEqual(self.tree.branch.base, revinfo.branch.base)
    self.assertEqual(self.tree2.branch.base, revinfo.branch.base)
    self.assertEqual(2, revinfo.revno)
    self.assertEqual(b'alt_r2', revinfo.rev_id)