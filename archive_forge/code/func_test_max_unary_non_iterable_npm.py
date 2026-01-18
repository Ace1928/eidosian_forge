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
def test_max_unary_non_iterable_npm(self):
    with self.assertTypingError():
        self.check_min_max_unary_non_iterable(max_usecase3)