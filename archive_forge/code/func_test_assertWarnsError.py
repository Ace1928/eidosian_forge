import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertWarnsError(self):
    """
        Test assertWarns failure when no warning is generated.
        """

    def normal(a):
        return a
    self.assertRaises(self.failureException, self.assertWarns, DeprecationWarning, 'Woo deprecated', __file__, normal, 123)