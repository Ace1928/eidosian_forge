from collections import namedtuple
import gc
import os
import operator
import sys
import weakref
import numpy as np
from numba.core import types, typing, errors, sigutils
from numba.core.types.abstract import _typecache
from numba.core.types.functions import _header_lead
from numba.core.typing.templates import make_overload_template
from numba import jit, njit, typeof
from numba.core.extending import (overload, register_model, models, unbox,
from numba.tests.support import TestCase, create_temp_module
from numba.tests.enum_usecases import Color, Shake, Shape
import unittest
from numba.np import numpy_support
from numba.core import types
def test_interning(self):
    a = types.Dummy('xyzzyx')
    code = a._code
    b = types.Dummy('xyzzyx')
    self.assertIs(b, a)
    wr = weakref.ref(a)
    del a
    gc.collect()
    c = types.Dummy('xyzzyx')
    self.assertIs(c, b)
    self.assertEqual(c._code, code)
    del b, c
    gc.collect()
    self.assertIs(wr(), None)
    d = types.Dummy('xyzzyx')
    self.assertNotEqual(d._code, code)