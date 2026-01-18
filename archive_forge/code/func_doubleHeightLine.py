from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def doubleHeightLine(self, top=True):
    if top:
        self.write(b'\x1b#3')
    else:
        self.write(b'\x1b#4')