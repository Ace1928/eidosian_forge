from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
@PygletView.method('v@')
def scrollWheel_(self, nsevent):
    x, y = getMousePosition(self, nsevent)
    scroll_x, scroll_y = getMouseDelta(nsevent)
    self._window.dispatch_event('on_mouse_scroll', x, y, scroll_x, scroll_y)