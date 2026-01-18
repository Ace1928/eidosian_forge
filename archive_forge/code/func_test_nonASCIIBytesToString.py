import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_nonASCIIBytesToString(self):
    """
        C{nativeString} raises a C{UnicodeError} if input bytes are not ASCII
        decodable.
        """
    self.assertRaises(UnicodeError, nativeString, b'\xff')