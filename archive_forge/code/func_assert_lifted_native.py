from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def assert_lifted_native(self, cres):
    jitloop = cres.lifted[0]
    [loopcres] = jitloop.overloads.values()
    self.assertTrue(loopcres.fndesc.native)