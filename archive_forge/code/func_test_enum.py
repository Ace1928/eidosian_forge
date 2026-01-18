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
def test_enum(self):
    tp_red = typeof(Color.red)
    self.assertEqual(tp_red, types.EnumMember(Color, types.intp))
    self.assertEqual(tp_red, typeof(Color.blue))
    tp_choc = typeof(Shake.chocolate)
    self.assertEqual(tp_choc, types.EnumMember(Shake, types.intp))
    self.assertEqual(tp_choc, typeof(Shake.mint))
    self.assertNotEqual(tp_choc, tp_red)
    tp_404 = typeof(RequestError.not_found)
    self.assertEqual(tp_404, types.IntEnumMember(RequestError, types.intp))
    self.assertEqual(tp_404, typeof(RequestError.internal_error))
    with self.assertRaises(ValueError) as raises:
        typeof(HeterogeneousEnum.red)
    self.assertEqual(str(raises.exception), 'Cannot type heterogeneous enum: got value types complex128, float64')