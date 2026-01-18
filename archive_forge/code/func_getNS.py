import struct
from cryptography.utils import int_to_bytes
from twisted.python.deprecate import deprecated
from twisted.python.versions import Version
def getNS(s, count=1):
    """
    get net string
    """
    ns = []
    c = 0
    for i in range(count):
        l, = struct.unpack('!L', s[c:c + 4])
        ns.append(s[c + 4:4 + l + c])
        c += 4 + l
    return tuple(ns) + (s[c:],)