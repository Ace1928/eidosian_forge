import gc
import weakref
import pytest
def test_clock_ended_callback(kivy_clock, clock_counter):
    counter2 = ClockCounter()
    counter_schedule = ClockCounter()
    kivy_clock.schedule_once(counter_schedule)
    event = kivy_clock.create_lifecycle_aware_trigger(clock_counter, counter2)
    event()
    kivy_clock.stop_clock()
    assert counter_schedule.counter == 0
    assert clock_counter.counter == 0
    assert counter2.counter == 1