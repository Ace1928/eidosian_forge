import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_dwim_spec_tag_that_looks_like_revno(self):
    self.tree.branch.tags.set_tag('3', b'r2')
    self.assertAsRevisionId(b'r2', '3')
    self.build_tree(['tree/b'])
    self.tree.add(['b'])
    self.tree.commit('b', rev_id=b'r3')
    self.assertAsRevisionId(b'r3', '3')