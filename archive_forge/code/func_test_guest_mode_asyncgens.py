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
@restore_unraisablehook()
def test_guest_mode_asyncgens() -> None:
    import sniffio
    record = set()

    async def agen(label: str) -> AsyncGenerator[int, None]:
        assert sniffio.current_async_library() == label
        try:
            yield 1
        finally:
            library = sniffio.current_async_library()
            with contextlib.suppress(trio.Cancelled):
                await sys.modules[library].sleep(0)
            record.add((label, library))

    async def iterate_in_aio() -> None:
        await agen('asyncio').asend(None)

    async def trio_main() -> None:
        task = asyncio.ensure_future(iterate_in_aio())
        done_evt = trio.Event()
        task.add_done_callback(lambda _: done_evt.set())
        with trio.fail_after(1):
            await done_evt.wait()
        await agen('trio').asend(None)
        gc_collect_harder()
    context = contextvars.copy_context()
    context.run(aiotrio_run, trio_main, host_uses_signal_set_wakeup_fd=True)
    assert record == {('asyncio', 'asyncio'), ('trio', 'trio')}