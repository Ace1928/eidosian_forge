from ctypes import c_void_p, c_bool
from pyglet.libs.darwin.cocoapy import ObjCClass, ObjCSubclass, send_super
from pyglet.libs.darwin.cocoapy import NSUInteger, NSUIntegerEncoding
from pyglet.libs.darwin.cocoapy import NSRectEncoding
@PygletToolWindow.method(b'@' + NSUIntegerEncoding + b'@@B')
def nextEventMatchingMask_untilDate_inMode_dequeue_(self, mask, date, mode, dequeue):
    if self.inLiveResize():
        from pyglet import app
        if app.event_loop is not None:
            app.event_loop.idle()
    event = send_super(self, 'nextEventMatchingMask:untilDate:inMode:dequeue:', mask, date, mode, dequeue, argtypes=[NSUInteger, c_void_p, c_void_p, c_bool])
    if event.value == None:
        return 0
    else:
        return event.value