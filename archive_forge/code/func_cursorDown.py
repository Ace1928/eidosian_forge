from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def cursorDown(self, n=1):
    assert n >= 1
    self.cursorPos.y = min(self.cursorPos.y + n, self.termSize.y - 1)
    self.write(b'\x1b[%dB' % (n,))