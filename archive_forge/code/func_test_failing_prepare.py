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
def test_failing_prepare(self):

    class A:

        def __array__(self):
            return np.zeros(1)

        def __array_prepare__(self, arr, context=None):
            raise RuntimeError
    a = A()
    assert_raises(RuntimeError, ncu.maximum, a, a)
    assert_raises(RuntimeError, ncu.maximum, a, a, where=False)