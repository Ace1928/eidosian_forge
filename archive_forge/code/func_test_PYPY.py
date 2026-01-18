import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_PYPY(self):
    """
        On PyPy, L{_PYPY} is True.
        """
    if 'PyPy' in sys.version:
        self.assertTrue(_PYPY)
    else:
        self.assertFalse(_PYPY)