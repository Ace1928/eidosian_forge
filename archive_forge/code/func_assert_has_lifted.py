from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def assert_has_lifted(self, jitted, loopcount):
    lifted = jitted.overloads[jitted.signatures[0]].lifted
    self.assertEqual(len(lifted), loopcount)