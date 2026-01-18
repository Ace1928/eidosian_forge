import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertWarnsWrongCategory(self):
    """
        Test assertWarns failure when the category is wrong.
        """

    def deprecated(a):
        warnings.warn('Foo deprecated', category=DeprecationWarning)
        return a
    self.assertRaises(self.failureException, self.assertWarns, UserWarning, 'Foo deprecated', __file__, deprecated, 123)