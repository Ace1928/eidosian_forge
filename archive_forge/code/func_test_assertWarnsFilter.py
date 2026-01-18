import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertWarnsFilter(self):
    """
        Test assertWarns on a warning filtered by default.
        """

    def deprecated(a):
        warnings.warn('Woo deprecated', category=PendingDeprecationWarning)
        return a
    r = self.assertWarns(PendingDeprecationWarning, 'Woo deprecated', __file__, deprecated, 123)
    self.assertEqual(r, 123)