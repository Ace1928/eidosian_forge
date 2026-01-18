from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
@PygletView.method('v@')
def otherMouseUp_(self, nsevent):
    x, y = getMousePosition(self, nsevent)
    buttons = mouse.MIDDLE
    modifiers = getModifiers(nsevent)
    self._window.dispatch_event('on_mouse_release', x, y, buttons, modifiers)