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
def isinstance_usecase_numba_types(a):
    if isinstance(a, typed.List):
        return 'typed list'
    elif isinstance(a, (types.int32, types.int64)):
        if isinstance(a, types.int32):
            return 'int32'
        else:
            return 'int64'
    elif isinstance(a, (types.float32, types.float64)):
        if isinstance(a, types.float32):
            return 'float32'
        elif isinstance(a, types.float64):
            return 'float64'
    elif isinstance(a, typed.Dict):
        return 'typed dict'
    else:
        return 'no match'