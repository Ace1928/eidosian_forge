import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def test_build_heterogeneous_set(self, flags=enable_pyobj_flags):
    pyfunc = set_literal_return_usecase((1, 2.0, 3j, 2))
    self.check(pyfunc)
    pyfunc = set_literal_return_usecase((2.0, 2))
    got, expected = self.check(pyfunc)
    self.assertIs(type(got.pop()), type(expected.pop()))