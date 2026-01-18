import numpy as np
import numpy
import unittest
from numba import njit, jit
from numba.core.errors import TypingError, UnsupportedError
from numba.core import ir
from numba.tests.support import TestCase, IRPreservingTestPipeline
def test_jit_multiple_closure_variables(self):
    self.run_jit_multiple_closure_variables(forceobj=True)