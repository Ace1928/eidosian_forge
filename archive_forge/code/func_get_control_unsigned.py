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
def get_control_unsigned(self, opname):
    op = getattr(operator, opname)

    def control_unsigned(a, b):
        tp = self.get_numpy_unsigned_upcast(a, b)
        return op(tp(a), tp(b))
    return control_unsigned