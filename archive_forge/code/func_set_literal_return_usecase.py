import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def set_literal_return_usecase(args):
    code = 'if 1:\n    def build_set():\n        return {%(initializer)s}\n    '
    return _build_set_literal_usecase(code, args)