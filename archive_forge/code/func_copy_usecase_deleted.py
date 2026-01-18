import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def copy_usecase_deleted(a, b):
    s = set(a)
    s.remove(b)
    ss = s.copy()
    s.pop()
    return (len(ss), list(ss))