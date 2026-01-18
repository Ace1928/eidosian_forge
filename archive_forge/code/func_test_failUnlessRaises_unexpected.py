import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failUnlessRaises_unexpected(self):
    try:
        self.failUnlessRaises(ValueError, self._raiseError, TypeError)
    except TypeError:
        self.fail("failUnlessRaises shouldn't re-raise unexpected exceptions")
    except self.failureException:
        pass
    else:
        self.fail("Expected exception wasn't raised. Should have failed")