import gc
import weakref
import pytest
def test_clock_stop_twice(kivy_clock, clock_counter):
    counter2 = ClockCounter()
    event = kivy_clock.create_lifecycle_aware_trigger(clock_counter, counter2)
    event()
    kivy_clock.stop_clock()
    assert clock_counter.counter == 0
    assert counter2.counter == 1
    kivy_clock.stop_clock()
    assert clock_counter.counter == 0
    assert counter2.counter == 1