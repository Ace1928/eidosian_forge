from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def begin_or_update_hover_event(self, win, *args):
    etype = 'update' if self.hover_event else 'begin'
    self.create_hover(win, etype)