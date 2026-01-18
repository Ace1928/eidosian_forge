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
def check_order(values):
    for i in range(len(values)):
        self.assertLessEqual(values[i], values[i])
        self.assertGreaterEqual(values[i], values[i])
        self.assertFalse(values[i] < values[i])
        self.assertFalse(values[i] > values[i])
        for j in range(i):
            self.assertLess(values[j], values[i])
            self.assertLessEqual(values[j], values[i])
            self.assertGreater(values[i], values[j])
            self.assertGreaterEqual(values[i], values[j])
            self.assertFalse(values[i] < values[j])
            self.assertFalse(values[i] <= values[j])
            self.assertFalse(values[j] > values[i])
            self.assertFalse(values[j] >= values[i])