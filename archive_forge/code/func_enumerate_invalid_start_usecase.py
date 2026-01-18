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
def enumerate_invalid_start_usecase():
    result = 0
    for i, j in enumerate((1.0, 2.5, 3.0), 3.14159):
        result += i * j
    return result