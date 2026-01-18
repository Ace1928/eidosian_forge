import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_intToBytes(self):
    """
        When L{intToBytes} is called with an integer, the result is an
        ASCII-encoded string representation of the number.
        """
    self.assertEqual(intToBytes(213), b'213')