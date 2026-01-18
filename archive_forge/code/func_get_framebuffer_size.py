from ctypes import *
import pyglet
from pyglet.window import BaseWindow
from pyglet.window import MouseCursor, DefaultMouseCursor
from pyglet.window import WindowException
from pyglet.event import EventDispatcher
from pyglet.canvas.cocoa import CocoaCanvas
from pyglet.libs.darwin import cocoapy, CGPoint, AutoReleasePool
from .systemcursor import SystemCursor
from .pyglet_delegate import PygletDelegate
from .pyglet_window import PygletWindow, PygletToolWindow
from .pyglet_view import PygletView
def get_framebuffer_size(self):
    view = self.context._nscontext.view()
    bounds = view.convertRectToBacking_(view.bounds()).size
    return (int(bounds.width), int(bounds.height))