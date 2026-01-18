import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
def test_isinstance_issue9125(self):
    pyfunc = invalid_isinstance_usecase_phi_nopropagate2
    cfunc = jit(nopython=True)(pyfunc)
    self.assertEqual(pyfunc(3), cfunc(3))