import inspect
import warnings
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet import defer, utils
from twisted.python import failure
from twisted.trial import itrial, util
from twisted.trial._synctest import FailTest, SkipTest, SynchronousTestCase
@defer.inlineCallbacks
def deferRunCleanups(self, ignored, result):
    """
        Run any scheduled cleanups and report errors (if any) to the result.
        object.
        """
    failures = []
    while len(self._cleanups) > 0:
        func, args, kwargs = self._cleanups.pop()
        try:
            yield func(*args, **kwargs)
        except Exception:
            failures.append(failure.Failure())
    for f in failures:
        result.addError(self, f)
        self._passed = False