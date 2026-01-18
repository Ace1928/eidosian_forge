import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertWarnsDifferentWarnings(self):
    """
        For now, assertWarns is unable to handle multiple different warnings,
        so it should raise an exception if it's the case.
        """

    def deprecated(a):
        warnings.warn('Woo deprecated', category=DeprecationWarning)
        warnings.warn('Another one', category=PendingDeprecationWarning)
    e = self.assertRaises(self.failureException, self.assertWarns, DeprecationWarning, 'Woo deprecated', __file__, deprecated, 123)
    self.assertEqual(str(e), "Can't handle different warnings")