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
def test_is_ellipsis(self):
    cfunc = njit((types.ellipsis, types.ellipsis))(self.op.is_usecase)
    self.assertTrue(cfunc(Ellipsis, Ellipsis))