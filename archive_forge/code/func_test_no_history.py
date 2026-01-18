import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_no_history(self):
    tree = self.make_branch_and_tree('tree3')
    self.assertRaises(errors.NoCommits, spec_in_history, 'last:', tree.branch)