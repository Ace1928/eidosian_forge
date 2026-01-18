import signal
from pyglet import app
from pyglet.app.base import PlatformEventLoop, EventLoop
from pyglet.libs.darwin import cocoapy, AutoReleasePool, ObjCSubclass, PyObjectEncoding, ObjCInstance, send_super, \
def nsapp_stop(self):
    """Used only for CocoaAlternateEventLoop"""
    self.NSApp.terminate_(None)