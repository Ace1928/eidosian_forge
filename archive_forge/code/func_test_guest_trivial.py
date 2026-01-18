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
def test_guest_trivial() -> None:

    async def trio_return(in_host: InHost) -> str:
        await trio.sleep(0)
        return 'ok'
    assert trivial_guest_run(trio_return) == 'ok'

    async def trio_fail(in_host: InHost) -> NoReturn:
        raise KeyError('whoopsiedaisy')
    with pytest.raises(KeyError, match='whoopsiedaisy'):
        trivial_guest_run(trio_fail)