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
def test_host_can_directly_wake_trio_task() -> None:

    async def trio_main(in_host: InHost) -> str:
        ev = trio.Event()
        in_host(ev.set)
        await ev.wait()
        return 'ok'
    assert trivial_guest_run(trio_main) == 'ok'