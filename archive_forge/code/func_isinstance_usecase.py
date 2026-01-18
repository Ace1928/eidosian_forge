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
def isinstance_usecase(a):
    if isinstance(a, (int, float)):
        if isinstance(a, int):
            return (a + 1, 'int')
        if isinstance(a, float):
            return (a + 2.0, 'float')
    elif isinstance(a, str):
        return (a + ', world!', 'str')
    elif isinstance(a, complex):
        return (a.imag, 'complex')
    elif isinstance(a, (tuple, list)):
        if isinstance(a, tuple):
            return 'tuple'
        else:
            return 'list'
    elif isinstance(a, set):
        return 'set'
    elif isinstance(a, bytes):
        return 'bytes'
    return 'no match'