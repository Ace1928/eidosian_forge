import contextlib
import inspect
import multiprocessing
import sys
import threading
from time import monotonic
import traceback
@contextlib.contextmanager
def decr_cm(self):
    with self._cond:
        self._active -= 1
        try:
            yield self._active
        finally:
            self._cond.notify_all()