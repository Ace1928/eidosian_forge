from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def selectCharacterSet(self, charSet, which):
    if which == G0:
        which = b'('
    elif which == G1:
        which = b')'
    else:
        raise ValueError("`which' argument to selectCharacterSet must be G0 or G1")
    if charSet == CS_UK:
        charSet = b'A'
    elif charSet == CS_US:
        charSet = b'B'
    elif charSet == CS_DRAWING:
        charSet = b'0'
    elif charSet == CS_ALTERNATE:
        charSet = b'1'
    elif charSet == CS_ALTERNATE_SPECIAL:
        charSet = b'2'
    else:
        raise ValueError("Invalid `charSet' argument to selectCharacterSet")
    self.write(b'\x1b' + which + charSet)