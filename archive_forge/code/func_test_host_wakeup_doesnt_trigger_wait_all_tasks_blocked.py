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
def test_host_wakeup_doesnt_trigger_wait_all_tasks_blocked() -> None:

    def set_deadline(cscope: trio.CancelScope, new_deadline: float) -> None:
        print(f'setting deadline {new_deadline}')
        cscope.deadline = new_deadline

    async def trio_main(in_host: InHost) -> str:

        async def sit_in_wait_all_tasks_blocked(watb_cscope: trio.CancelScope) -> None:
            with watb_cscope:
                await trio.testing.wait_all_tasks_blocked(cushion=9999)
                raise AssertionError('wait_all_tasks_blocked should *not* return normally, only by cancellation.')
            assert watb_cscope.cancelled_caught

        async def get_woken_by_host_deadline(watb_cscope: trio.CancelScope) -> None:
            with trio.CancelScope() as cscope:
                print('scheduling stuff to happen')

                class InstrumentHelper(Instrument):

                    def __init__(self) -> None:
                        self.primed = False

                    def before_io_wait(self, timeout: float) -> None:
                        print(f'before_io_wait({timeout})')
                        if timeout == 9999:
                            assert not self.primed
                            in_host(lambda: set_deadline(cscope, 1000000000.0))
                            self.primed = True

                    def after_io_wait(self, timeout: float) -> None:
                        if self.primed:
                            print('instrument triggered')
                            in_host(lambda: cscope.cancel())
                            trio.lowlevel.remove_instrument(self)
                trio.lowlevel.add_instrument(InstrumentHelper())
                await trio.sleep_forever()
            assert cscope.cancelled_caught
            watb_cscope.cancel()
        async with trio.open_nursery() as nursery:
            watb_cscope = trio.CancelScope()
            nursery.start_soon(sit_in_wait_all_tasks_blocked, watb_cscope)
            await trio.testing.wait_all_tasks_blocked()
            nursery.start_soon(get_woken_by_host_deadline, watb_cscope)
        return 'ok'
    assert trivial_guest_run(trio_main) == 'ok'