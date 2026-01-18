import time
from math import inf
import pytest
from trio import sleep
from ... import _core
from .. import wait_all_tasks_blocked
from .._mock_clock import MockClock
from .tutil import slow
def test_mock_clock() -> None:
    REAL_NOW = 123.0
    c = MockClock()
    c._real_clock = lambda: REAL_NOW
    repr(c)
    assert c.rate == 0
    assert c.current_time() == 0
    c.jump(1.2)
    assert c.current_time() == 1.2
    with pytest.raises(ValueError, match="^time can't go backwards$"):
        c.jump(-1)
    assert c.current_time() == 1.2
    assert c.deadline_to_sleep_time(1.1) == 0
    assert c.deadline_to_sleep_time(1.2) == 0
    assert c.deadline_to_sleep_time(1.3) > 999999
    with pytest.raises(ValueError, match='^rate must be >= 0$'):
        c.rate = -1
    assert c.rate == 0
    c.rate = 2
    assert c.current_time() == 1.2
    REAL_NOW += 1
    assert c.current_time() == 3.2
    assert c.deadline_to_sleep_time(3.1) == 0
    assert c.deadline_to_sleep_time(3.2) == 0
    assert c.deadline_to_sleep_time(4.2) == 0.5
    c.rate = 0.5
    assert c.current_time() == 3.2
    assert c.deadline_to_sleep_time(3.1) == 0
    assert c.deadline_to_sleep_time(3.2) == 0
    assert c.deadline_to_sleep_time(4.2) == 2.0
    c.jump(0.8)
    assert c.current_time() == 4.0
    REAL_NOW += 1
    assert c.current_time() == 4.5
    c2 = MockClock(rate=3)
    assert c2.rate == 3
    assert c2.current_time() < 10