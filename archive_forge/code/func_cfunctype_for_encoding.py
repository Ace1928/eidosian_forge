import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def cfunctype_for_encoding(encoding):
    if encoding in cfunctype_table:
        return cfunctype_table[encoding]
    typecodes = {b'c': c_char, b'i': c_int, b's': c_short, b'l': c_long, b'q': c_longlong, b'C': c_ubyte, b'I': c_uint, b'S': c_ushort, b'L': c_ulong, b'Q': c_ulonglong, b'f': c_float, b'd': c_double, b'B': c_bool, b'v': None, b'*': c_char_p, b'@': c_void_p, b'#': c_void_p, b':': c_void_p, NSPointEncoding: NSPoint, NSSizeEncoding: NSSize, NSRectEncoding: NSRect, NSRangeEncoding: NSRange, PyObjectEncoding: py_object}
    argtypes = []
    for code in parse_type_encoding(encoding):
        if code in typecodes:
            argtypes.append(typecodes[code])
        elif code[0:1] == b'^' and code[1:] in typecodes:
            argtypes.append(POINTER(typecodes[code[1:]]))
        else:
            raise Exception('unknown type encoding: ' + code)
    cfunctype = CFUNCTYPE(*argtypes)
    cfunctype_table[encoding] = cfunctype
    return cfunctype