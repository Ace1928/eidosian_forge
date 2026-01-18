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
def test_minval_maxval(self):
    self.assertEqual(types.int8.maxval, 127)
    self.assertEqual(types.int8.minval, -128)
    self.assertEqual(types.uint8.maxval, 255)
    self.assertEqual(types.uint8.minval, 0)
    self.assertEqual(types.int64.maxval, (1 << 63) - 1)
    self.assertEqual(types.int64.minval, -(1 << 63))
    self.assertEqual(types.uint64.maxval, (1 << 64) - 1)
    self.assertEqual(types.uint64.minval, 0)