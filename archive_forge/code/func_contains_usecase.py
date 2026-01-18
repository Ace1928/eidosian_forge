import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def contains_usecase(a, b):
    s = set(a)
    l = []
    for v in b:
        l.append(v in s)
    return l