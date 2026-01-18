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
def test_array_in_struct(self):
    dll = CDLL(_ctypes_test.__file__)

    class Test2(Structure):
        _fields_ = [('data', c_ubyte * 16)]

    class Test3AParent(Structure):
        _fields_ = [('data', c_float * 2)]

    class Test3A(Test3AParent):
        _fields_ = [('more_data', c_float * 2)]

    class Test3B(Structure):
        _fields_ = [('data', c_double * 2)]

    class Test3C(Structure):
        _fields_ = [('data', c_double * 4)]

    class Test3D(Structure):
        _fields_ = [('data', c_double * 8)]

    class Test3E(Structure):
        _fields_ = [('data', c_double * 9)]
    s = Test2()
    expected = 0
    for i in range(16):
        s.data[i] = i
        expected += i
    func = dll._testfunc_array_in_struct2
    func.restype = c_int
    func.argtypes = (Test2,)
    result = func(s)
    self.assertEqual(result, expected)
    for i in range(16):
        self.assertEqual(s.data[i], i)
    s = Test3A()
    s.data[0] = 3.14159
    s.data[1] = 2.71828
    s.more_data[0] = -3.0
    s.more_data[1] = -2.0
    expected = 3.14159 + 2.71828 - 3.0 - 2.0
    func = dll._testfunc_array_in_struct3A
    func.restype = c_double
    func.argtypes = (Test3A,)
    result = func(s)
    self.assertAlmostEqual(result, expected, places=6)
    self.assertAlmostEqual(s.data[0], 3.14159, places=6)
    self.assertAlmostEqual(s.data[1], 2.71828, places=6)
    self.assertAlmostEqual(s.more_data[0], -3.0, places=6)
    self.assertAlmostEqual(s.more_data[1], -2.0, places=6)
    StructCtype = namedtuple('StructCtype', ['cls', 'cfunc1', 'cfunc2', 'items'])
    structs_to_test = [StructCtype(Test3B, dll._testfunc_array_in_struct3B, dll._testfunc_array_in_struct3B_set_defaults, 2), StructCtype(Test3C, dll._testfunc_array_in_struct3C, dll._testfunc_array_in_struct3C_set_defaults, 4), StructCtype(Test3D, dll._testfunc_array_in_struct3D, dll._testfunc_array_in_struct3D_set_defaults, 8), StructCtype(Test3E, dll._testfunc_array_in_struct3E, dll._testfunc_array_in_struct3E_set_defaults, 9)]
    for sut in structs_to_test:
        s = sut.cls()
        expected = 0
        for i in range(sut.items):
            float_i = float(i)
            s.data[i] = float_i
            expected += float_i
        func = sut.cfunc1
        func.restype = c_double
        func.argtypes = (sut.cls,)
        result = func(s)
        self.assertEqual(result, expected)
        for i in range(sut.items):
            self.assertEqual(s.data[i], float(i))
        func = sut.cfunc2
        func.restype = sut.cls
        result = func()
        for i in range(sut.items):
            self.assertEqual(result.data[i], float(i + 1))