import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_get_async_param_deprecation(self):
    """
        L{twisted.python.compat._get_async_param} raises a deprecation
        warning if async keyword argument is passed.
        """
    self.assertEqual(_get_async_param(isAsync=None, **{'async': False}), False)
    currentWarnings = self.flushWarnings(offendingFunctions=[self.test_get_async_param_deprecation])
    self.assertEqual(currentWarnings[0]['message'], "'async' keyword argument is deprecated, please use isAsync")