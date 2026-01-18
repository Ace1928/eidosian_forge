from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def reportCursorPosition(self):
    d = defer.Deferred()
    self._cursorReports.append(d)
    self.write(b'\x1b[6n')
    return d