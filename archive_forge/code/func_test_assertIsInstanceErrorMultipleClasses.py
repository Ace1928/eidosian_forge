import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertIsInstanceErrorMultipleClasses(self):
    """
        Test an error with assertIsInstance and multiple classes.
        """
    A = type('A', (object,), {})
    B = type('B', (object,), {})
    C = type('C', (object,), {})
    a = A()
    self.assertRaises(self.failureException, self.assertIsInstance, a, (B, C))