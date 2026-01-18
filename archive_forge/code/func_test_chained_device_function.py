from numba.tests.support import override_config
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import itertools
import re
import unittest
def test_chained_device_function(self):
    debug_opts = itertools.product(*[(True, False)] * 3)
    for kernel_debug, f1_debug, f2_debug in debug_opts:
        with self.subTest(kernel_debug=kernel_debug, f1_debug=f1_debug, f2_debug=f2_debug):
            self._test_chained_device_function(kernel_debug, f1_debug, f2_debug)