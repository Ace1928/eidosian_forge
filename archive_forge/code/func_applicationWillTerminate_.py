import signal
from pyglet import app
from pyglet.app.base import PlatformEventLoop, EventLoop
from pyglet.libs.darwin import cocoapy, AutoReleasePool, ObjCSubclass, PyObjectEncoding, ObjCInstance, send_super, \
@_AppDelegate.method('v@')
def applicationWillTerminate_(self, notification):
    self._pyglet_loop.is_running = False
    self._pyglet_loop.has_exit = True