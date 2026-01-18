import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failIf_matches_assertNot(self):
    asserts = prefixedMethods(unittest.SynchronousTestCase, 'assertNot')
    failIfs = prefixedMethods(unittest.SynchronousTestCase, 'failIf')
    self.assertEqual(sorted(asserts, key=self._name), sorted(failIfs, key=self._name))