import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertWarnsOnMethod(self):
    """
        Test assertWarns works when used on an instance method.
        """

    class Warn:

        def deprecated(self, a):
            warnings.warn('Bar deprecated', category=DeprecationWarning)
            return a
    w = Warn()
    r = self.assertWarns(DeprecationWarning, 'Bar deprecated', __file__, w.deprecated, 321)
    self.assertEqual(r, 321)
    r = self.assertWarns(DeprecationWarning, 'Bar deprecated', __file__, w.deprecated, 321)
    self.assertEqual(r, 321)