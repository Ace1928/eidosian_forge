import signal
from pyglet import app
from pyglet.app.base import PlatformEventLoop, EventLoop
from pyglet.libs.darwin import cocoapy, AutoReleasePool, ObjCSubclass, PyObjectEncoding, ObjCInstance, send_super, \
def term_received(*args):
    if self.timer:
        self.timer.invalidate()
        self.timer = None
    self.nsapp_stop()