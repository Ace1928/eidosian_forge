import pytest
from kivy.compat import isclose
from kivy.input import MotionEvent
def create_dummy_motion_event(self):
    return DummyMotionEvent('dummy', 'dummy1', (0, 0))