from kivy.config import Config
from kivy.logger import Logger
from kivy.input import providers
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def auto_calibrate(self, sx, sy, size):
    from kivy.core.window import Window as W
    WIDTH, HEIGHT = size
    xratio = WIDTH / W.width
    yratio = HEIGHT / W.height
    xoffset = -W.left / W.width
    yoffset = -(HEIGHT - W.top - W.height) / W.height
    sx = sx * xratio + xoffset
    sy = sy * yratio + yoffset
    return (sx, sy)