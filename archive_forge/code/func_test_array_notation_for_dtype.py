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
def test_array_notation_for_dtype(self):

    def check(arrty, scalar, ndim, layout):
        self.assertIs(arrty.dtype, scalar)
        self.assertEqual(arrty.ndim, ndim)
        self.assertEqual(arrty.layout, layout)
    scalar = types.int32
    dtyped = types.DType(scalar)
    check(dtyped[:], scalar, 1, 'A')
    check(dtyped[::1], scalar, 1, 'C')
    check(dtyped[:, :], scalar, 2, 'A')
    check(dtyped[:, ::1], scalar, 2, 'C')
    check(dtyped[::1, :], scalar, 2, 'F')