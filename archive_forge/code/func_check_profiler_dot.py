import cProfile as profiler
import os
import pstats
import subprocess
import sys
import numpy as np
from numba import jit
from numba.tests.support import needs_blas, expected_failure_py312
import unittest
def check_profiler_dot(self, pyfunc):
    """
        Make sure the jit-compiled function shows up in the profile stats
        as a regular Python function.
        """
    a = np.arange(16, dtype=np.float32)
    b = np.arange(16, dtype=np.float32)
    cfunc = jit(nopython=True)(pyfunc)
    cfunc(a, b)
    p = profiler.Profile()
    p.enable()
    try:
        cfunc(a, b)
    finally:
        p.disable()
    stats = pstats.Stats(p).strip_dirs()
    code = pyfunc.__code__
    expected_key = (os.path.basename(code.co_filename), code.co_firstlineno, code.co_name)
    self.assertIn(expected_key, stats.stats)