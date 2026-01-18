from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def l(self, proto, handler, buf):
    try:
        modes = [int(mode) for mode in buf.split(b';')]
    except ValueError:
        handler.unhandledControlSequence(b'\x1b[' + buf + 'l')
    else:
        handler.resetModes(modes)