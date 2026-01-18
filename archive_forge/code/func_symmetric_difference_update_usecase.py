import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def symmetric_difference_update_usecase(a, b):
    s = set(a)
    s.symmetric_difference_update(set(b))
    return list(s)