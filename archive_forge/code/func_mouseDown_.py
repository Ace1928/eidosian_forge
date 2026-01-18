from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
@PygletView.method('v@')
def mouseDown_(self, nsevent):
    x, y = getMousePosition(self, nsevent)
    buttons = mouse.LEFT
    modifiers = getModifiers(nsevent)
    self._window.dispatch_event('on_mouse_press', x, y, buttons, modifiers)