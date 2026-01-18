from __future__ import annotations
import contextvars
import queue as stdlib_queue
import re
import sys
import threading
import time
import weakref
from functools import partial
from typing import (
import pytest
import sniffio
from .. import (
from .._core._tests.test_ki import ki_self
from .._core._tests.tutil import slow
from .._threads import (
from ..testing import wait_all_tasks_blocked
def test_from_thread_run_during_shutdown() -> None:
    save = []
    record = []

    async def agen(token: _core.TrioToken | None) -> AsyncGenerator[None, None]:
        try:
            yield
        finally:
            with _core.CancelScope(shield=True):
                try:
                    await to_thread_run_sync(partial(from_thread_run, sleep, 0, trio_token=token))
                except _core.RunFinishedError:
                    record.append('finished')
                else:
                    record.append('clean')

    async def main(use_system_task: bool) -> None:
        save.append(agen(_core.current_trio_token() if use_system_task else None))
        await save[-1].asend(None)
    _core.run(main, True)
    _core.run(main, False)
    assert record == ['finished', 'clean']