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
def test_guest_mode_sniffio_integration() -> None:
    from sniffio import current_async_library, thread_local as sniffio_library

    async def trio_main(in_host: InHost) -> str:

        async def synchronize() -> None:
            """Wait for all in_host() calls issued so far to complete."""
            evt = trio.Event()
            in_host(evt.set)
            await evt.wait()
        in_host(partial(setattr, sniffio_library, 'name', 'nullio'))
        await synchronize()
        assert current_async_library() == 'trio'
        record = []
        in_host(lambda: record.append(current_async_library()))
        await synchronize()
        assert record == ['nullio']
        assert current_async_library() == 'trio'
        return 'ok'
    try:
        assert trivial_guest_run(trio_main) == 'ok'
    finally:
        sniffio_library.name = None