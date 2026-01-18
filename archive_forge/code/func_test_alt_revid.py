import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_alt_revid(self):
    self.assertInHistoryIs(1, b'r1', 'before:revid:alt_r2')