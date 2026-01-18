from __future__ import annotations
from random import randrange
from typing import Any, Callable, TextIO, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import interfaces, utils
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
def postClassCleanup(self):
    """
        Called by L{unittest.TestCase} after the last test in a C{TestCase}
        subclass. Ensures the reactor is clean by murdering the threadpool,
        catching any pending
        L{DelayedCall<twisted.internet.base.DelayedCall>}s, open sockets etc.
        """
    selectables = self._cleanReactor()
    calls = self._cleanPending()
    if selectables or calls:
        aggregate = DirtyReactorAggregateError(calls, selectables)
        self.result.addError(self.test, Failure(aggregate))
    self._cleanThreads()