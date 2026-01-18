import array
from collections import namedtuple
import enum
import mmap
import typing as py_typing
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaValueError, NumbaTypeError
from numba.misc.special import typeof
from numba.core.dispatcher import OmittedArg
from numba._dispatcher import compute_fingerprint
from numba.tests.support import TestCase, skip_unless_cffi, tag
from numba.tests.test_numpy_support import ValueTypingTestBase
from numba.tests.ctypes_usecases import *
from numba.tests.enum_usecases import *
from numba.np import numpy_support
def test_enum_class(self):
    tp_color = typeof(Color)
    self.assertEqual(tp_color, types.EnumClass(Color, types.intp))
    tp_shake = typeof(Shake)
    self.assertEqual(tp_shake, types.EnumClass(Shake, types.intp))
    self.assertNotEqual(tp_shake, tp_color)
    tp_shape = typeof(Shape)
    self.assertEqual(tp_shape, types.IntEnumClass(Shape, types.intp))
    tp_error = typeof(RequestError)
    self.assertEqual(tp_error, types.IntEnumClass(RequestError, types.intp))
    self.assertNotEqual(tp_error, tp_shape)
    with self.assertRaises(ValueError) as raises:
        typeof(HeterogeneousEnum)
    self.assertEqual(str(raises.exception), 'Cannot type heterogeneous enum: got value types complex128, float64')