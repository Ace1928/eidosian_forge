from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def deleteCharacter(self, n=1):
    self.write(b'\x1b[%dP' % (n,))