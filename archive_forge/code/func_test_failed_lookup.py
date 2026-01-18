import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_failed_lookup(self):
    self.assertRaises(errors.NoSuchTag, self.get_in_history, 'tag:some-random-tag')