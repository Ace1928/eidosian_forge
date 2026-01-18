import platform
from platform import architecture as _architecture
import struct
import sys
import unittest
from ctypes.test import need_symbol
from ctypes import (CDLL, Array, Structure, Union, POINTER, sizeof, byref, alignment,
from ctypes.util import find_library
from struct import calcsize
import _ctypes_test
from collections import namedtuple
from test import support
@unittest.skipIf(True, 'Test disabled for now - see bpo-16575/bpo-16576')
def test_bitfield_by_value(self):

    class Test6(Structure):
        _fields_ = [('A', c_int, 1), ('B', c_int, 2), ('C', c_int, 3), ('D', c_int, 2)]
    test6 = Test6()
    test6.A = 1
    test6.B = 3
    test6.C = 7
    test6.D = 3
    dll = CDLL(_ctypes_test.__file__)
    with self.assertRaises(TypeError) as ctx:
        func = dll._testfunc_bitfield_by_value1
        func.restype = c_long
        func.argtypes = (Test6,)
        result = func(test6)
    self.assertEqual(ctx.exception.args[0], 'item 1 in _argtypes_ passes a struct/union with a bitfield by value, which is unsupported.')
    func = dll._testfunc_bitfield_by_reference1
    func.restype = c_long
    func.argtypes = (POINTER(Test6),)
    result = func(byref(test6))
    self.assertEqual(result, -4)
    self.assertEqual(test6.A, 0)
    self.assertEqual(test6.B, 0)
    self.assertEqual(test6.C, 0)
    self.assertEqual(test6.D, 0)

    class Test7(Structure):
        _fields_ = [('A', c_uint, 1), ('B', c_uint, 2), ('C', c_uint, 3), ('D', c_uint, 2)]
    test7 = Test7()
    test7.A = 1
    test7.B = 3
    test7.C = 7
    test7.D = 3
    func = dll._testfunc_bitfield_by_reference2
    func.restype = c_long
    func.argtypes = (POINTER(Test7),)
    result = func(byref(test7))
    self.assertEqual(result, 14)
    self.assertEqual(test7.A, 0)
    self.assertEqual(test7.B, 0)
    self.assertEqual(test7.C, 0)
    self.assertEqual(test7.D, 0)

    class Test8(Union):
        _fields_ = [('A', c_int, 1), ('B', c_int, 2), ('C', c_int, 3), ('D', c_int, 2)]
    test8 = Test8()
    with self.assertRaises(TypeError) as ctx:
        func = dll._testfunc_bitfield_by_value2
        func.restype = c_long
        func.argtypes = (Test8,)
        result = func(test8)
    self.assertEqual(ctx.exception.args[0], 'item 1 in _argtypes_ passes a union by value, which is unsupported.')