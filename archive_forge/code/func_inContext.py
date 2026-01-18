from __future__ import annotations
from threading import Thread, current_thread
from typing import Any, Callable, List, Optional, TypeVar
from typing_extensions import ParamSpec, Protocol, TypedDict
from twisted._threads import pool as _pool
from twisted.python import context, log
from twisted.python.deprecate import deprecated
from twisted.python.failure import Failure
from twisted.python.versions import Version
def inContext() -> None:
    try:
        result = inContext.theWork()
        ok = True
    except BaseException:
        result = Failure()
        ok = False
    inContext.theWork = None
    if inContext.onResult is not None:
        inContext.onResult(ok, result)
        inContext.onResult = None
    elif not ok:
        log.err(result)