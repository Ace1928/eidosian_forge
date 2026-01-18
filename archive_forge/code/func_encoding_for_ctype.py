from ctypes import *
import sys, platform, struct
def encoding_for_ctype(vartype):
    typecodes = {c_char: b'c', c_int: b'i', c_short: b's', c_long: b'l', c_longlong: b'q', c_ubyte: b'C', c_uint: b'I', c_ushort: b'S', c_ulong: b'L', c_ulonglong: b'Q', c_float: b'f', c_double: b'd', c_bool: b'B', c_char_p: b'*', c_void_p: b'@', py_object: PyObjectEncoding}
    return typecodes.get(vartype, b'?')