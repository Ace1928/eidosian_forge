import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_lazyByteSliceOffset(self):
    """
        L{lazyByteSlice} called with some bytes and an offset returns a
        semantically equal version of these bytes starting at the given offset.
        """
    data = b'123XYZ'
    self.assertEqual(bytes(lazyByteSlice(data, 2)), data[2:])