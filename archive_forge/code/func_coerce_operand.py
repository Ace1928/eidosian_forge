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
def coerce_operand(self, op, numba_type):
    if hasattr(op, 'dtype'):
        return numba_type.cast_python_value(op)
    elif numba_type in types.unsigned_domain:
        return abs(int(op.real))
    elif numba_type in types.integer_domain:
        return int(op.real)
    elif numba_type in types.real_domain:
        return float(op.real)
    else:
        return op