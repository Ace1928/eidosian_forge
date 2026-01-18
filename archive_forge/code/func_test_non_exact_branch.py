import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_non_exact_branch(self):
    self.assertRaises(errors.NotBranchError, self.get_in_history, 'branch:tree2/a')