import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_3openTextMode(self):
    """
        A file opened via 'io.open' in text mode accepts and returns text.
        """
    with open(self.mktemp(), 'w') as f:
        self.assertEqual(ioType(f), str)