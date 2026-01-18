import sys
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types, errors, utils
from numba.tests.support import (captured_stdout, TestCase, EnableNRTStatsMixin)
def make_print_closure(x):

    def print_closure():
        return x
    return jit(nopython=True)(x)