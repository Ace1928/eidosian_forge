import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_dotted_decimal(self):
    self.assertInHistoryIs(None, b'alt_r2', '1.1.1')
    self.assertInvalid('1.1.123')