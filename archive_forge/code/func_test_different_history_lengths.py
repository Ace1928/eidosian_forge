import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_different_history_lengths(self):
    self.tree2.commit('three', rev_id=b'r3')
    self.assertInHistoryIs(3, b'r3', 'revno:3:tree2')
    self.assertInHistoryIs(3, b'r3', 'revno:-1:tree2')