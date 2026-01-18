import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_callDeprecatedCallsFunction(self):
    """
        L{callDeprecated} actually calls the callable passed to it, and
        forwards the result.
        """
    result = self.callDeprecated(self.version, oldMethod, 'foo')
    self.assertEqual('foo', result)