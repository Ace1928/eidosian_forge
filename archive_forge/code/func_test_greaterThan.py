import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_greaterThan(self):
    """
        L{cmp} returns 1 if its first argument is bigger than its second.
        """
    self.assertEqual(cmp(4, 0), 1)
    self.assertEqual(cmp(b'z', b'a'), 1)