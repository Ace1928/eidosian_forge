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
def test_guest_mode_on_asyncio() -> None:

    async def trio_main() -> str:
        print('trio_main!')
        to_trio, from_aio = trio.open_memory_channel[int](float('inf'))
        from_trio: asyncio.Queue[int] = asyncio.Queue()
        aio_task = asyncio.ensure_future(aio_pingpong(from_trio, to_trio))
        await trio.sleep(0)
        from_trio.put_nowait(0)
        async for n in from_aio:
            print(f'trio got: {n}')
            from_trio.put_nowait(n + 1)
            if n >= 10:
                aio_task.cancel()
                return 'trio-main-done'
        raise AssertionError('should never be reached')

    async def aio_pingpong(from_trio: asyncio.Queue[int], to_trio: MemorySendChannel[int]) -> None:
        print('aio_pingpong!')
        try:
            while True:
                n = await from_trio.get()
                print(f'aio got: {n}')
                to_trio.send_nowait(n + 1)
        except asyncio.CancelledError:
            raise
        except:
            traceback.print_exc()
            raise
    assert aiotrio_run(trio_main, host_uses_signal_set_wakeup_fd=True) == 'trio-main-done'
    assert aiotrio_run(trio_main, pass_not_threadsafe=False, host_uses_signal_set_wakeup_fd=True) == 'trio-main-done'