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
def test_guest_is_initialized_when_start_returns() -> None:
    trio_token = None
    record = []

    async def trio_main(in_host: InHost) -> str:
        record.append('main task ran')
        await trio.sleep(0)
        assert trio.lowlevel.current_trio_token() is trio_token
        return 'ok'

    def after_start() -> None:
        assert record == []
        nonlocal trio_token
        trio_token = trio.lowlevel.current_trio_token()
        trio_token.run_sync_soon(record.append, 'run_sync_soon cb ran')

        @trio.lowlevel.spawn_system_task
        async def early_task() -> None:
            record.append('system task ran')
            await trio.sleep(0)
    res = trivial_guest_run(trio_main, in_host_after_start=after_start)
    assert res == 'ok'
    assert set(record) == {'system task ran', 'main task ran', 'run_sync_soon cb ran'}

    class BadClock:

        def start_clock(self) -> NoReturn:
            raise ValueError('whoops')

    def after_start_never_runs() -> None:
        pytest.fail("shouldn't get here")
    with pytest.raises(trio.TrioInternalError):
        trivial_guest_run(trio_main, clock=BadClock(), in_host_after_start=after_start_never_runs)