from __future__ import annotations
from random import randrange
from typing import Any, Callable, TextIO, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import interfaces, utils
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
def postCaseCleanup(self):
    """
        Called by L{unittest.TestCase} after a test to catch any logged errors
        or pending L{DelayedCall<twisted.internet.base.DelayedCall>}s.
        """
    calls = self._cleanPending()
    if calls:
        aggregate = DirtyReactorAggregateError(calls)
        self.result.addError(self.test, Failure(aggregate))
        return False
    return True