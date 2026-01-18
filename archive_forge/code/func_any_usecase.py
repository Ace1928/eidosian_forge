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
def any_usecase(x, y):
    if x == None and y == None:
        return any([])
    elif x == None:
        return any([y])
    elif y == None:
        return any([x])
    else:
        return any([x, y])