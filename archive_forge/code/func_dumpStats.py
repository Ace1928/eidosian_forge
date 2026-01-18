from __future__ import annotations
from threading import Thread, current_thread
from typing import Any, Callable, List, Optional, TypeVar
from typing_extensions import ParamSpec, Protocol, TypedDict
from twisted._threads import pool as _pool
from twisted.python import context, log
from twisted.python.deprecate import deprecated
from twisted.python.failure import Failure
from twisted.python.versions import Version
def dumpStats(self) -> None:
    """
        Dump some plain-text informational messages to the log about the state
        of this L{ThreadPool}.
        """
    log.msg(f'waiters: {self.waiters}')
    log.msg(f'workers: {self.working}')
    log.msg(f'total: {self.threads}')