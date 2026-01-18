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
def test_guest_mode_ki() -> None:
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    async def trio_main(in_host: InHost) -> None:
        with pytest.raises(KeyboardInterrupt):
            signal_raise(signal.SIGINT)
        in_host(partial(signal_raise, signal.SIGINT))
        await trio.sleep(10)
    with pytest.raises(KeyboardInterrupt) as excinfo:
        trivial_guest_run(trio_main)
    assert excinfo.value.__context__ is None
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    final_exc = KeyError('whoa')

    async def trio_main_raising(in_host: InHost) -> NoReturn:
        in_host(partial(signal_raise, signal.SIGINT))
        raise final_exc
    with pytest.raises(KeyboardInterrupt) as excinfo:
        trivial_guest_run(trio_main_raising)
    assert excinfo.value.__context__ is final_exc
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler