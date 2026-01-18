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
def run_binop_complex(self, pyfunc, flags=force_pyobj_flags):
    x_operands = [-1.1 + 0.3j, 0.0 + 0j, 1.1j]
    y_operands = [-1.5 - 0.7j, 0.8j, 2.1 - 2j]
    types_list = [(types.complex64, types.complex64), (types.complex128, types.complex128)]
    self.run_test_floats(pyfunc, x_operands, y_operands, types_list, flags=flags)