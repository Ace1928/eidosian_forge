from __future__ import annotations
import ctypes
import ctypes.util
import sys
import traceback
from functools import partial
from itertools import count
from threading import Lock, Thread
from typing import Any, Callable, Generic, TypeVar
import outcome
def start_thread_soon(self, fn: Callable[[], RetT], deliver: Callable[[outcome.Outcome[RetT]], object], name: str | None=None) -> None:
    worker: WorkerThread[RetT]
    try:
        worker, _ = self._idle_workers.popitem()
    except KeyError:
        worker = WorkerThread(self)
    worker._job = (fn, deliver, name)
    worker._worker_lock.release()