import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertEqual_basic(self):
    self._testEqualPair('cat', 'cat')
    self._testUnequalPair('cat', 'dog')
    self._testEqualPair([1], [1])
    self._testUnequalPair([1], 'orange')