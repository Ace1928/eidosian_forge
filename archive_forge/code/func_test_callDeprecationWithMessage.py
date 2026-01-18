import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_callDeprecationWithMessage(self):
    """
        L{callDeprecated} can take a message argument used to check the warning
        emitted.
        """
    self.callDeprecated((self.version, 'newMethod'), oldMethodReplaced, 1)