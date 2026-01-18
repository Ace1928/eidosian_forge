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
@unittest.skipIf(sc is None, 'Only run if SciPy >= 0.19 is installed')
def test_missing_function(self):
    with self.assertRaises(ValueError) as raises:
        get_cython_function_address('scipy.special.cython_special', 'foo')
    msg = "No function 'foo' found in __pyx_capi__ of 'scipy.special.cython_special'"
    self.assertEqual(msg, str(raises.exception))