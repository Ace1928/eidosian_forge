import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failUnlessRaises_noException(self):
    returnValue = 3
    try:
        self.failUnlessRaises(ValueError, lambda: returnValue)
    except self.failureException as e:
        self.assertEqual(str(e), 'ValueError not raised (3 returned)')
    else:
        self.fail('Exception not raised. Should have failed')