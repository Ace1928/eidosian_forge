import signal
import sys
import types
from typing import Any, TypeVar
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
from .. import _core
from .._core._tests.tutil import (
from .._util import (
from ..testing import wait_all_tasks_blocked
def test_module_metadata_is_fixed_up() -> None:
    import trio
    import trio.testing
    assert trio.Cancelled.__module__ == 'trio'
    assert trio.open_nursery.__module__ == 'trio'
    assert trio.abc.Stream.__module__ == 'trio.abc'
    assert trio.lowlevel.wait_task_rescheduled.__module__ == 'trio.lowlevel'
    assert trio.testing.trio_test.__module__ == 'trio.testing'
    assert trio.lowlevel.ParkingLot.__init__.__module__ == 'trio.lowlevel'
    assert trio.abc.Stream.send_all.__module__ == 'trio.abc'
    assert trio.Cancelled.__name__ == 'Cancelled'
    assert trio.Cancelled.__qualname__ == 'Cancelled'
    assert trio.abc.SendStream.send_all.__name__ == 'send_all'
    assert trio.abc.SendStream.send_all.__qualname__ == 'SendStream.send_all'
    assert trio.to_thread.__name__ == 'trio.to_thread'
    assert trio.to_thread.run_sync.__name__ == 'run_sync'
    assert trio.to_thread.run_sync.__qualname__ == 'run_sync'