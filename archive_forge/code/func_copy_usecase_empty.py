import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def copy_usecase_empty(a):
    s = set(a)
    s.clear()
    ss = s.copy()
    s.add(a[0])
    return (len(ss), list(ss))