import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def test_array_too_many_args(self):

    class A:

        def __array__(self, dtype, context):
            return np.zeros(1)
    a = A()
    assert_raises_regex(TypeError, '2 required positional', np.sum, a)