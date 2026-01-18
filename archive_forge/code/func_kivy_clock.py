import pytest
import gc
import weakref
import time
import os.path
@pytest.fixture()
def kivy_clock():
    from kivy.context import Context
    from kivy.clock import ClockBase
    context = Context(init=False)
    context['Clock'] = ClockBase()
    context.push()
    from kivy.clock import Clock
    Clock._max_fps = 0
    try:
        Clock.start_clock()
        yield Clock
        Clock.stop_clock()
    finally:
        context.pop()