import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_dwim_spec_revno(self):
    self.assertInHistoryIs(2, b'r2', '2')
    self.assertAsRevisionId(b'alt_r2', '1.1.1')