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
def test_array_notation(self):

    def check(arrty, scalar, ndim, layout):
        self.assertIs(arrty.dtype, scalar)
        self.assertEqual(arrty.ndim, ndim)
        self.assertEqual(arrty.layout, layout)

    def check_index_error(callable):
        with self.assertRaises(KeyError) as raises:
            callable()
        self.assertIn('Can only index numba types with slices with no start or stop, got', str(raises.exception))
    scalar = types.int32
    check(scalar[:], scalar, 1, 'A')
    check(scalar[::1], scalar, 1, 'C')
    check(scalar[:, :], scalar, 2, 'A')
    check(scalar[:, ::1], scalar, 2, 'C')
    check(scalar[::1, :], scalar, 2, 'F')
    check_index_error(lambda: scalar[0])
    check_index_error(lambda: scalar[:, 4])
    check_index_error(lambda: scalar[::1, 1:])
    check_index_error(lambda: scalar[:2])
    check_index_error(lambda: list(scalar))