import inspect
import sys
import types
import warnings
from os.path import normcase
from warnings import catch_warnings, simplefilter
from incremental import Version
from twisted.python import deprecate
from twisted.python.deprecate import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.python.test import deprecatedattributes
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.python.deprecate import deprecatedModuleAttribute
from incremental import Version
from twisted.python import deprecate
from twisted.python import deprecate
def test_getDeprecationWarningString(self):
    """
        L{getDeprecationWarningString} returns a string that tells us that a
        callable was deprecated at a certain released version of Twisted.
        """
    version = Version('Twisted', 8, 0, 0)
    self.assertEqual(getDeprecationWarningString(self.test_getDeprecationWarningString, version), '%s.DeprecationWarningsTests.test_getDeprecationWarningString was deprecated in Twisted 8.0.0' % (__name__,))