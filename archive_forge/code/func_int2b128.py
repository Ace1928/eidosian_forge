import copy
import struct
from io import BytesIO
from twisted.internet import protocol
from twisted.persisted import styles
from twisted.python import log
from twisted.python.compat import iterbytes
from twisted.python.reflect import fullyQualifiedName
def int2b128(integer, stream):
    if integer == 0:
        stream(b'\x00')
        return
    assert integer > 0, 'can only encode positive integers'
    while integer:
        stream(bytes((integer & 127,)))
        integer = integer >> 7