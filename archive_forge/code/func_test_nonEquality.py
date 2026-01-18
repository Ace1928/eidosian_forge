import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_nonEquality(self):
    """
        Instances of a class that is decorated by C{comparable} support
        inequality comparisons.
        """
    self.assertFalse(Comparable(1) != Comparable(1))
    self.assertTrue(Comparable(2) != Comparable(1))