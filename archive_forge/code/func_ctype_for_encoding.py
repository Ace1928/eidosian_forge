import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def ctype_for_encoding(self, encoding):
    """Return ctypes type for an encoded Objective-C type."""
    if encoding in self.typecodes:
        return self.typecodes[encoding]
    elif encoding[0:1] == b'^' and encoding[1:] in self.typecodes:
        return POINTER(self.typecodes[encoding[1:]])
    elif encoding[0:1] == b'^' and encoding[1:] in [CGImageEncoding, NSZoneEncoding]:
        return c_void_p
    elif encoding[0:1] == b'r' and encoding[1:] in self.typecodes:
        return self.typecodes[encoding[1:]]
    elif encoding[0:2] == b'r^' and encoding[2:] in self.typecodes:
        return POINTER(self.typecodes[encoding[2:]])
    else:
        raise Exception('unknown encoding for %s: %s' % (self.name, encoding))