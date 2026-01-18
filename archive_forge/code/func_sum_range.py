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
@njit
def sum_range(sz, start=0):
    tmp = range(sz)
    ret = sum(tmp, start)
    return (sum(tmp, start=start), ret)