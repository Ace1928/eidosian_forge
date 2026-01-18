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
def test_list_type_getitem(self):
    for listty in (types.int64, types.Array(types.float64, 1, 'C')):
        l_int = types.List(listty)
        self.assertTrue(isinstance(l_int, types.List))
        self.assertTrue(isinstance(l_int[0], type(listty)))