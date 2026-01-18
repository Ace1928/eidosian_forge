from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.lang import Builder
from kivy.base import EventLoop
from kivy.weakproxy import WeakProxy
from time import sleep
def move_frames(self, t):
    for i in range(t):
        EventLoop.idle()