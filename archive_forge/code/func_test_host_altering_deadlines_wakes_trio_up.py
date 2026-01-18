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
def test_host_altering_deadlines_wakes_trio_up() -> None:

    def set_deadline(cscope: trio.CancelScope, new_deadline: float) -> None:
        cscope.deadline = new_deadline

    async def trio_main(in_host: InHost) -> str:
        with trio.CancelScope() as cscope:
            in_host(lambda: set_deadline(cscope, -inf))
            await trio.sleep_forever()
        assert cscope.cancelled_caught
        with trio.CancelScope() as cscope:
            in_host(lambda: set_deadline(cscope, 1000000.0))
            in_host(lambda: set_deadline(cscope, -inf))
            await trio.sleep(999)
        assert cscope.cancelled_caught
        return 'ok'
    assert trivial_guest_run(trio_main) == 'ok'