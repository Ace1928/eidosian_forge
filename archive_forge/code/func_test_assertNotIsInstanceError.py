import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertNotIsInstanceError(self):
    """
        Test an error with assertNotIsInstance.
        """
    A = type('A', (object,), {})
    a = A()
    error = self.assertRaises(self.failureException, self.assertNotIsInstance, a, A)
    self.assertEqual(str(error), f'{a!r} is an instance of {A}')