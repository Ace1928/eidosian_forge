from kivy.logger import Logger
from functools import partial
from collections import deque
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
def depack(self, args):
    self.sx, self.sy, self.a, self.X, self.Y, sw, sh, sd, self.A, self.m, self.r = args
    self.Y = -self.Y
    if self.shape is None:
        self.shape = ShapeRect()
        self.shape.width = sw
        self.shape.height = sh
    self.sy = 1 - self.sy
    super().depack(args)