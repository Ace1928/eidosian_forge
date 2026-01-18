import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertIsInstanceCustomMessage(self):
    """
        If L{TestCase.assertIsInstance} is passed a custom message as its 3rd
        argument, the message is included in the failure exception raised when
        the assertion fails.
        """
    exc = self.assertRaises(self.failureException, self.assertIsInstance, 3, str, 'Silly assertion')
    self.assertIn('Silly assertion', str(exc))