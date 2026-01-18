import cProfile as profiler
import os
import pstats
import subprocess
import sys
import numpy as np
from numba import jit
from numba.tests.support import needs_blas, expected_failure_py312
import unittest
def test_issue_3229(self):
    code = "if 1:\n            import cProfile as profiler\n            p = profiler.Profile()\n            p.enable()\n\n            from numba.tests.npyufunc.test_dufunc import TestDUFunc\n            t = TestDUFunc('test_npm_call')\n            t.test_npm_call()\n\n            p.disable()\n            "
    subprocess.check_call([sys.executable, '-c', code])