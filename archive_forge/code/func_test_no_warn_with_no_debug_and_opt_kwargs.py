import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.support import linux_only, override_config
from numba.core.errors import NumbaPerformanceWarning
import warnings
def test_no_warn_with_no_debug_and_opt_kwargs(self):
    with warnings.catch_warnings(record=True) as w:
        cuda.jit()
    self.assertEqual(len(w), 0)