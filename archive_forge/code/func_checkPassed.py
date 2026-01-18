import inspect
import sys
import types
import warnings
from os.path import normcase
from warnings import catch_warnings, simplefilter
from incremental import Version
from twisted.python import deprecate
from twisted.python.deprecate import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.python.test import deprecatedattributes
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.python.deprecate import deprecatedModuleAttribute
from incremental import Version
from twisted.python import deprecate
from twisted.python import deprecate
def checkPassed(self, func, *args, **kw):
    """
        Test an invocation of L{passed} with the given function, arguments, and
        keyword arguments.

        @param func: A function whose argspec to pass to L{_passed}.
        @type func: A callable.

        @param args: The arguments which could be passed to L{func}.

        @param kw: The keyword arguments which could be passed to L{func}.

        @return: L{_passed}'s return value
        @rtype: L{dict}
        """
    return _passedSignature(inspect.signature(func), args, kw)