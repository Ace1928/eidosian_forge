from __future__ import annotations
import threading
import time
from contextlib import contextmanager
from queue import Queue
from typing import TYPE_CHECKING, Iterator, NoReturn
import pytest
from .. import _thread_cache
from .._thread_cache import ThreadCache, start_thread_soon
from .tutil import gc_collect_harder, slow
@slow
def test_idle_threads_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_thread_cache, 'IDLE_TIMEOUT', 0.0001)
    q: Queue[threading.Thread] = Queue()
    start_thread_soon(lambda: None, lambda _: q.put(threading.current_thread()))
    seen_thread = q.get()
    time.sleep(1)
    assert not seen_thread.is_alive()