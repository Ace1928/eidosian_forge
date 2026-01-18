import time
from math import inf
import pytest
from trio import sleep
from ... import _core
from .. import wait_all_tasks_blocked
from .._mock_clock import MockClock
from .tutil import slow
def test_mock_clock_autojump_preset() -> None:
    mock_clock = MockClock(autojump_threshold=0.1)
    mock_clock.autojump_threshold = 0.01
    real_start = time.perf_counter()
    _core.run(sleep, 10000, clock=mock_clock)
    assert time.perf_counter() - real_start < 1