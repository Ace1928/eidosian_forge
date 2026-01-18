import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_consume_no_stream(self):
    f_arr = ForeignArray(cuda.device_array(10))
    c_arr = cuda.as_cuda_array(f_arr)
    self.assertEqual(c_arr.stream, 0)