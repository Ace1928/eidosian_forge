from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
@PygletView.method('v@')
def cursorUpdate_(self, nsevent):
    self._window._mouse_in_window = True
    if not self._window._mouse_exclusive:
        self._window.set_mouse_platform_visible()