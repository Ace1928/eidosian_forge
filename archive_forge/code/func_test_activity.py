from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def test_activity(self):
    if not self.disable_on_activity:
        return False
    for touch in EventLoop.touches:
        if touch.__class__.__name__ == 'KineticMotionEvent':
            continue
        if touch.__class__ != MouseMotionEvent:
            return True
    return False