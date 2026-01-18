import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertIsNotError(self):
    """
        L{assertIsNot} fails if two objects are identical.
        """
    a = MockEquality('first')
    self.assertRaises(self.failureException, self.assertIsNot, a, a)