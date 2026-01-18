import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_invalid_line(self):
    spec = RevisionSpec.from_string('annotate:annotate-tree/file1:q')
    e = self.assertRaises(InvalidRevisionSpec, spec.as_revision_id, self.tree.branch)
    self.assertContainsRe(str(e), "Requested revision: \\'annotate:annotate-tree/file1:q\\' does not exist in branch: .*\nNo such line: q")