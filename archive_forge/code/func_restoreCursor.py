from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def restoreCursor(self):
    self.cursorPos = self._savedCursorPos
    del self._savedCursorPos
    self.write(b'\x1b8')