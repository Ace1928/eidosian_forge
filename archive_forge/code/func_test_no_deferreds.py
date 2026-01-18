import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_no_deferreds(self):
    marker = object()
    result, errors = _spinner.trap_unhandled_errors(lambda: marker)
    self.assertEqual([], errors)
    self.assertIs(marker, result)