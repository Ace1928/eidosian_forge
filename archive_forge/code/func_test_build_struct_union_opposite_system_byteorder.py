import sys, unittest, struct, math, ctypes
from binascii import hexlify
from ctypes import *
def test_build_struct_union_opposite_system_byteorder(self):
    if sys.byteorder == 'little':
        _Structure = BigEndianStructure
        _Union = BigEndianUnion
    else:
        _Structure = LittleEndianStructure
        _Union = LittleEndianUnion

    class S1(_Structure):
        _fields_ = [('a', c_byte), ('b', c_byte)]

    class U1(_Union):
        _fields_ = [('s1', S1), ('ab', c_short)]

    class S2(_Structure):
        _fields_ = [('u1', U1), ('c', c_byte)]