import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertEqual_custom(self):
    x = MockEquality('first')
    y = MockEquality('second')
    z = MockEquality('first')
    self._testEqualPair(x, x)
    self._testEqualPair(x, z)
    self._testUnequalPair(x, y)
    self._testUnequalPair(y, z)