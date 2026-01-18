import unicodedata
from pyglet.window import key
from pyglet.libs.darwin.cocoapy import ObjCClass, ObjCSubclass, ObjCInstance
from pyglet.libs.darwin.cocoapy import PyObjectEncoding, send_super
from pyglet.libs.darwin.cocoapy import CFSTR, cfstring_to_string, cf
@PygletTextView.method('v@')
def insertNewline_(self, sender):
    event = NSApplication.sharedApplication().currentEvent()
    chars = event.charactersIgnoringModifiers()
    ch = chr(chars.characterAtIndex_(0))
    if ch == u'\r':
        self._window.dispatch_event('on_text', u'\r')