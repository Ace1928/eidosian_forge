import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_as_revision_id_r2(self):
    self.assertAsRevisionId(b'r2', 'annotate:annotate-tree/file1:1')