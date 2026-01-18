import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.support import linux_only, override_config
from numba.core.errors import NumbaPerformanceWarning
import warnings
def test_no_warn_on_debug_and_no_opt(self):
    with warnings.catch_warnings(record=True) as w:
        cuda.jit(debug=True, opt=False)
    self.assertEqual(len(w), 0)