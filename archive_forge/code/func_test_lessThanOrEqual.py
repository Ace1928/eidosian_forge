import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_lessThanOrEqual(self):
    """
        Instances of a class that is decorated by C{comparable} support
        less-than-or-equal comparisons.
        """
    self.assertTrue(Comparable(3) <= Comparable(3))
    self.assertTrue(Comparable(0) <= Comparable(3))
    self.assertFalse(Comparable(2) <= Comparable(0))