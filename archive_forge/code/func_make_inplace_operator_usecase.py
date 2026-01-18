import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def make_inplace_operator_usecase(self, op):
    code = 'if 1:\n        def inplace_operator_usecase(a, b):\n            sa = a\n            sb = b\n            sc = sa\n            sc %(op)s sb\n            return list(sc), list(sa)\n        ' % dict(op=op)
    return compile_function('inplace_operator_usecase', code, globals())