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
def run_arith_binop(self, pyfunc, opname, samples, expected_type=int, force_type=lambda x: x, **assertPreciseEqualArgs):
    self.run_binary(pyfunc, self.get_control_signed(opname), samples, self.signed_pairs, expected_type, force_type=force_type, **assertPreciseEqualArgs)
    self.run_binary(pyfunc, self.get_control_unsigned(opname), samples, self.unsigned_pairs, expected_type, force_type=force_type, **assertPreciseEqualArgs)