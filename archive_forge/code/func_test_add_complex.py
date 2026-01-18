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
def test_add_complex(self, flags=force_pyobj_flags):
    pyfunc = self.op.add_usecase
    x_operands = [1 + 0j, 1j, -1 - 1j]
    y_operands = x_operands
    types_list = [(types.complex64, types.complex64), (types.complex128, types.complex128)]
    self.run_test_floats(pyfunc, x_operands, y_operands, types_list, flags=flags)