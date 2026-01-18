import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def assertNativeString(self, original, expected):
    """
        Raise an exception indicating a failed test if the output of
        C{nativeString(original)} is unequal to the expected string, or is not
        a native string.
        """
    self.assertEqual(nativeString(original), expected)
    self.assertIsInstance(nativeString(original), str)