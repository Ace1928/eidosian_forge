import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_alt_no_parents(self):
    new_tree = self.make_branch_and_tree('new_tree')
    new_tree.commit('first', rev_id=b'new_r1')
    self.tree.branch.fetch(new_tree.branch, b'new_r1')
    self.assertInHistoryIs(0, b'null:', 'before:revid:new_r1')