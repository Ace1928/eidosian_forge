import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertWarnsWrongFile(self):
    """
        If the warning emitted by a function refers to a different file than is
        passed to C{assertWarns}, C{failureException} is raised.
        """

    def deprecated(a):
        warnings.warn('Foo deprecated', category=DeprecationWarning, stacklevel=2)
    self.assertRaises(self.failureException, self.assertWarns, DeprecationWarning, 'Foo deprecated', __file__, deprecated, 123)