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
def test_getattrPassthrough(self):
    """
        Getting a normal attribute on a L{twisted.python.deprecate._ModuleProxy}
        retrieves the underlying attribute's value, and raises C{AttributeError}
        if a non-existent attribute is accessed.
        """
    proxy = self._makeProxy(SOME_ATTRIBUTE='hello')
    self.assertIs(proxy.SOME_ATTRIBUTE, 'hello')
    self.assertRaises(AttributeError, getattr, proxy, 'DOES_NOT_EXIST')