from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
@PygletView.method('v@')
def keyUp_(self, nsevent):
    symbol = getSymbol(nsevent)
    modifiers = getModifiers(nsevent)
    self._window.dispatch_event('on_key_release', symbol, modifiers)