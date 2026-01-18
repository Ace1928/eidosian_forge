from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def on_frame(self, controller):
    frame = controller.frame()
    _LEAP_QUEUE.append(frame)