import copy
import struct
from io import BytesIO
from twisted.internet import protocol
from twisted.persisted import styles
from twisted.python import log
from twisted.python.compat import iterbytes
from twisted.python.reflect import fullyQualifiedName
def b1282int(st):
    """
    Convert an integer represented as a base 128 string into an L{int}.

    @param st: The integer encoded in a byte string.
    @type st: L{bytes}

    @return: The integer value extracted from the byte string.
    @rtype: L{int}
    """
    e = 1
    i = 0
    for char in iterbytes(st):
        n = ord(char)
        i += n * e
        e <<= 7
    return i