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
def test_default_args_starargs_and_keyonly(self):
    spec = [('x', int32), ('y', int32), ('z', int32), ('args', types.UniTuple(int32, 2)), ('a', int32)]
    with self.assertRaises(errors.UnsupportedError) as raises:
        jitclass(TestClass2, spec)
    msg = 'VAR_POSITIONAL argument type unsupported'
    self.assertIn(msg, str(raises.exception))