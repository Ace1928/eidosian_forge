import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def test_invalid_number(self):
    try:
        int('X')
    except ValueError as e:
        self.assertInvalid('revno:X', extra='\n' + str(e))
    else:
        self.fail()