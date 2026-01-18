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
def all_usecase(x, y):
    if x == None and y == None:
        return all([])
    elif x == None:
        return all([y])
    elif y == None:
        return all([x])
    else:
        return all([x, y])