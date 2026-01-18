import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
def test_flags_no_nrt(self):

    @register_jitable(_nrt=False)
    def foo(n):
        return np.arange(n)

    def bar(n):
        return foo(n)
    self.assertEqual(bar(3).tolist(), [0, 1, 2])
    cbar = jit(nopython=True)(bar)
    with self.assertRaises(errors.TypingError) as raises:
        cbar(2)
    msg = 'Only accept returning of array passed into the function as argument'
    self.assertIn(msg, str(raises.exception))