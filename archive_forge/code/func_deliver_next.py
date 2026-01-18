from __future__ import annotations
import signal
from collections import OrderedDict
from contextlib import contextmanager
from typing import TYPE_CHECKING
import trio
from ._util import ConflictDetector, is_main_thread, signal_raise
def deliver_next() -> None:
    if self._pending:
        signum, _ = self._pending.popitem(last=False)
        try:
            signal_raise(signum)
        finally:
            deliver_next()