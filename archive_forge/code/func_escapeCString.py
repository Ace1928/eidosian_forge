import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def escapeCString(src):
    dst = ''
    for c in src:
        if c in '\\"':
            dst += '\\' + c
        elif ord(c) < 32:
            if c == '\n':
                dst += '\\n'
            elif c == '\r':
                dst += '\\r'
            elif c == '\x07':
                dst += '\\a'
            elif c == '\x08':
                dst += '\\b'
            elif c == '\x0c':
                dst += '\\f'
            elif c == '\t':
                dst += '\\t'
            elif c == '\x0b':
                dst += '\\v'
            else:
                dst += '\\%03o' % ord(c)
        else:
            dst += c
    return dst