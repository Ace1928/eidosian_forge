import numpy as np
from io import StringIO
from numba import cuda, float32, float64, int32, intp
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import (skip_on_cudasim, skip_with_nvdisasm,
@skip_with_nvdisasm('Missing nvdisasm exception only generated when it is not present')
def test_inspect_sass_nvdisasm_missing(self):

    @cuda.jit((float32[::1],))
    def f(x):
        x[0] = 0
    with self.assertRaises(RuntimeError) as raises:
        f.inspect_sass()
    self.assertIn('nvdisasm has not been found', str(raises.exception))