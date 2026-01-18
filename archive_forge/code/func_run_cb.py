from __future__ import annotations
import threading
from collections import deque
from typing import TYPE_CHECKING, Callable, NoReturn, Tuple
import attrs
from .. import _core
from .._util import NoPublicConstructor, final
from ._wakeup_socketpair import WakeupSocketpair
def run_cb(job: Job) -> None:
    sync_fn, args = job
    try:
        sync_fn(*args)
    except BaseException as exc:

        async def kill_everything(exc: BaseException) -> NoReturn:
            raise exc
        try:
            _core.spawn_system_task(kill_everything, exc)
        except RuntimeError:
            parent_nursery = _core.current_task().parent_nursery
            if parent_nursery is None:
                raise AssertionError('Internal error: `parent_nursery` should never be `None`') from exc
            parent_nursery.start_soon(kill_everything, exc)