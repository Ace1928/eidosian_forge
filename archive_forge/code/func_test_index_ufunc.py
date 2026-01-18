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
def test_index_ufunc(self):
    """
        Check Numpy ufunc on an Index object.
        """
    i = Index(np.int32([42, 8, -5]))
    cfunc = jit(nopython=True)(npyufunc_usecase)
    ii = cfunc(i)
    self.assertIsInstance(ii, Index)
    self.assertPreciseEqual(ii._data, np.cos(np.sin(i._data)))