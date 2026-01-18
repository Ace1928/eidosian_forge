from the multitouch provider.
from kivy.base import EventLoop
from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def on_mouse_motion(self, win, x, y, modifiers):
    nx, ny = win.to_normalized_pos(x, y)
    ny = 1.0 - ny
    if self.current_drag:
        touch = self.current_drag
        touch.move([nx, ny])
        touch.update_graphics(win)
        self.waiting_event.append(('update', touch))
    elif self.alt_touch is not None and 'alt' not in modifiers:
        is_double_tap = 'shift' in modifiers
        self.create_touch(win, nx, ny, is_double_tap, True, [])