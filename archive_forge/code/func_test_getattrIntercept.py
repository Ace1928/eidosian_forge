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
def test_getattrIntercept(self):
    """
        Getting an attribute marked as being deprecated on
        L{twisted.python.deprecate._ModuleProxy} results in calling the
        deprecated wrapper's C{get} method.
        """
    proxy = self._makeProxy()
    _deprecatedAttributes = object.__getattribute__(proxy, '_deprecatedAttributes')
    _deprecatedAttributes['foo'] = _MockDeprecatedAttribute(42)
    self.assertEqual(proxy.foo, 42)