from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def remove_touch(self, win, touch):
    if touch.id in self.touches:
        del self.touches[touch.id]
        touch.update_time_end()
        self.waiting_event.append(('end', touch))
        touch.clear_graphics(win)