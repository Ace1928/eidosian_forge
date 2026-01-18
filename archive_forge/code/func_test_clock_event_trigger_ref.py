import gc
import weakref
import pytest
def test_clock_event_trigger_ref(kivy_clock):
    value = None

    class Counter:

        def call(self, *args, **kwargs):
            nonlocal value
            value = 42
    event = kivy_clock.create_trigger(Counter().call)
    gc.collect()
    event()
    kivy_clock.tick()
    assert value is None
    kivy_clock.schedule_once(Counter().call)
    event()
    kivy_clock.tick()
    assert value is None
    event = kivy_clock.create_trigger(Counter().call, release_ref=False)
    gc.collect()
    event()
    kivy_clock.tick()
    assert value == 42