from pyglet.libs.darwin.cocoapy import ObjCClass, ObjCSubclass, ObjCInstance
from pyglet.libs.darwin.cocoapy import NSApplicationDidHideNotification
from pyglet.libs.darwin.cocoapy import NSApplicationDidUnhideNotification
from pyglet.libs.darwin.cocoapy import send_super, get_selector
from pyglet.libs.darwin.cocoapy import PyObjectEncoding
from pyglet.libs.darwin.cocoapy import quartz
from .systemcursor import SystemCursor
@PygletDelegate.method('v@')
def applicationDidUnhide_(self, notification):
    if self._window._mouse_exclusive and quartz.CGCursorIsVisible():
        SystemCursor.unhide()
        SystemCursor.hide()
        pass
    self._window.dispatch_event('on_show')