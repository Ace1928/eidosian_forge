import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def gen_impl(fn):

    def impl():
        try:
            op()
        except err:
            fn()
    return impl