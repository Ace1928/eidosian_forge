import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertDictEqual(self):
    """
        L{twisted.trial.unittest.TestCase} supports the C{assertDictEqual}
        method inherited from the standard library in Python 2.7.
        """
    self.assertDictEqual({'a': 1}, {'a': 1})