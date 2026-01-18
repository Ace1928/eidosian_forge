from __future__ import annotations
import threading
from collections import deque
from typing import TYPE_CHECKING, Callable, NoReturn, Tuple
import attrs
from .. import _core
from .._util import NoPublicConstructor, final
from ._wakeup_socketpair import WakeupSocketpair
def run_all_bounded() -> None:
    for _ in range(len(self.queue)):
        run_cb(self.queue.popleft())
    for job in list(self.idempotent_queue):
        del self.idempotent_queue[job]
        run_cb(job)