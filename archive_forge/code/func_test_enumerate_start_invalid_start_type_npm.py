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
def test_enumerate_start_invalid_start_type_npm(self):
    pyfunc = enumerate_invalid_start_usecase
    with self.assertRaises(errors.TypingError) as raises:
        jit((), **no_pyobj_flags)(pyfunc)
    msg = 'Only integers supported as start value in enumerate'
    self.assertIn(msg, str(raises.exception))