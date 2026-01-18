import ctypes
import itertools
import pickle
import random
import typing as pt
import unittest
from collections import OrderedDict
import numpy as np
from numba import (boolean, deferred_type, float32, float64, int16, int32,
from numba.core import errors, types
from numba.core.dispatcher import Dispatcher
from numba.core.errors import LoweringError, TypingError
from numba.core.runtime.nrt import MemInfo
from numba.experimental import jitclass
from numba.experimental.jitclass import _box
from numba.experimental.jitclass.base import JitClassType
from numba.tests.support import MemoryLeakMixin, TestCase, skip_if_typeguard
from numba.tests.support import skip_unless_scipy
def test_c_structure(self):
    spec = OrderedDict()
    spec['a'] = int32
    spec['b'] = int16
    spec['c'] = float64

    @jitclass(spec)
    class Struct(object):

        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c
    st = Struct(43981, 239, 3.1415)

    class CStruct(ctypes.Structure):
        _fields_ = [('a', ctypes.c_int32), ('b', ctypes.c_int16), ('c', ctypes.c_double)]
    ptr = ctypes.c_void_p(_box.box_get_dataptr(st))
    cstruct = ctypes.cast(ptr, ctypes.POINTER(CStruct))[0]
    self.assertEqual(cstruct.a, st.a)
    self.assertEqual(cstruct.b, st.b)
    self.assertEqual(cstruct.c, st.c)