import unittest
from ctypes.test import need_symbol
import test.support
def test_noctypes_argtype(self):
    import _ctypes_test
    from ctypes import CDLL, c_void_p, ArgumentError
    func = CDLL(_ctypes_test.__file__)._testfunc_p_p
    func.restype = c_void_p
    self.assertRaises(TypeError, setattr, func, 'argtypes', (object,))

    class Adapter:

        def from_param(cls, obj):
            return None
    func.argtypes = (Adapter(),)
    self.assertEqual(func(None), None)
    self.assertEqual(func(object()), None)

    class Adapter:

        def from_param(cls, obj):
            return obj
    func.argtypes = (Adapter(),)
    self.assertRaises(ArgumentError, func, object())
    self.assertEqual(func(c_void_p(42)), 42)

    class Adapter:

        def from_param(cls, obj):
            raise ValueError(obj)
    func.argtypes = (Adapter(),)
    self.assertRaises(ArgumentError, func, 99)