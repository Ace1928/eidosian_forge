from __future__ import annotations
import asyncio
import contextlib
import contextvars
import queue
import signal
import socket
import sys
import threading
import time
import traceback
import warnings
from functools import partial
from math import inf
from typing import (
import pytest
from outcome import Outcome
import trio
import trio.testing
from trio.abc import Instrument
from ..._util import signal_raise
from .tutil import gc_collect_harder, restore_unraisablehook
def run_sync_soon_threadsafe(fn: Callable[[], object]) -> None:
    nonlocal todo
    if host_thread is threading.current_thread():
        crash = partial(pytest.fail, 'run_sync_soon_threadsafe called from host thread')
        todo.put(('run', crash))
    todo.put(('run', fn))