from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def create_hover(self, win, etype):
    nx, ny = win.to_normalized_pos(*win.mouse_pos)
    nx /= win._density
    ny /= win._density
    args = (nx, ny)
    hover = self.hover_event
    if hover:
        hover.move(args)
    else:
        self.hover_event = hover = MouseMotionEvent(self.device, self.create_event_id(), args, type_id='hover')
    if etype == 'end':
        hover.update_time_end()
        self.hover_event = None
    self.waiting_event.append((etype, hover))