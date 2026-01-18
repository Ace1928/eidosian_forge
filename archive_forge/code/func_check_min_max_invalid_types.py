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
def check_min_max_invalid_types(self, pyfunc, flags=forceobj_flags):
    cfunc = jit((types.int32, types.Dummy('list')), **flags)(pyfunc)
    cfunc(1, [1])