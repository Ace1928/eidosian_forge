from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
@PygletView.method('v@')
def mouseEntered_(self, nsevent):
    x, y = getMousePosition(self, nsevent)
    self._window._mouse_in_window = True
    self._window.dispatch_event('on_mouse_enter', x, y)