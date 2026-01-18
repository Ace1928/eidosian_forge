import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def reflect_simple(sa, sb):
    sa.add(42)
    sa.update(sb)
    return (sa, len(sa), len(sb))