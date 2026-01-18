from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def test_lift_gen1(self):
    self.check_lift_generator_ok(lift_gen1, (types.intp,), (123,))