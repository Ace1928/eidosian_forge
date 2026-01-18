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
def test_globals(self, flags=forceobj_flags):
    pyfunc = globals_usecase
    cfunc = jit((), **flags)(pyfunc)
    g = cfunc()
    self.assertIs(g, globals())