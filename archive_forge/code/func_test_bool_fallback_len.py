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
def test_bool_fallback_len(self):

    class NoBoolHasLen:

        def __init__(self, val):
            self.val = val

        def __len__(self):
            return self.val

        def get_bool(self):
            return bool(self)
    py_class = NoBoolHasLen
    jitted_class = jitclass([('val', types.int64)])(py_class)
    py_class_0_bool = py_class(0).get_bool()
    py_class_2_bool = py_class(2).get_bool()
    jitted_class_0_bool = jitted_class(0).get_bool()
    jitted_class_2_bool = jitted_class(2).get_bool()
    self.assertEqual(py_class_0_bool, jitted_class_0_bool)
    self.assertEqual(py_class_2_bool, jitted_class_2_bool)
    self.assertEqual(type(py_class_0_bool), type(jitted_class_0_bool))
    self.assertEqual(type(py_class_2_bool), type(jitted_class_2_bool))