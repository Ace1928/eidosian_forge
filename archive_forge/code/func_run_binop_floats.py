import copy
import itertools
import operator
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, utils, errors
from numba.core.types.functions import _header_lead
from numba.tests.support import TestCase, tag, needs_blas
from numba.tests.matmul_usecase import (matmul_usecase, imatmul_usecase,
def run_binop_floats(self, pyfunc, flags=force_pyobj_flags):
    x_operands = [-1.1, 0.0, 1.1]
    y_operands = [-1.5, 0.8, 2.1]
    types_list = [(types.float32, types.float32), (types.float64, types.float64)]
    self.run_test_floats(pyfunc, x_operands, y_operands, types_list, flags=flags)