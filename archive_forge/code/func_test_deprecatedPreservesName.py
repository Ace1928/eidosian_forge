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
def test_deprecatedPreservesName(self):
    """
        The decorated function has the same name as the original.
        """
    version = Version('Twisted', 8, 0, 0)
    dummy = deprecated(version)(dummyCallable)
    self.assertEqual(dummyCallable.__name__, dummy.__name__)
    self.assertEqual(fullyQualifiedName(dummyCallable), fullyQualifiedName(dummy))