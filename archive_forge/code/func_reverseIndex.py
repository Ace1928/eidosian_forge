from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def reverseIndex(self):
    self.cursorPos.y = max(self.cursorPos.y - 1, 0)
    self.write(b'\x1bM')