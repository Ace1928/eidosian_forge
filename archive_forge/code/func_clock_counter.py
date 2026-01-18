import gc
import weakref
import pytest
@pytest.fixture()
def clock_counter():
    yield ClockCounter()