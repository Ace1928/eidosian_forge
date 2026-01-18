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
def run_binop_bools(self, pyfunc, flags=force_pyobj_flags):
    x_operands = [False, False, True, True]
    y_operands = [False, True, False, True]
    types_list = [(types.boolean, types.boolean)]
    self.run_test_ints(pyfunc, x_operands, y_operands, types_list, flags=flags)