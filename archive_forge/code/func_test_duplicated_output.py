import numpy as np
from collections import namedtuple
from numba import void, int32, float32, float64
from numba import guvectorize
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
import warnings
from numba.core.errors import NumbaPerformanceWarning
from numba.tests.support import override_config
def test_duplicated_output(self):

    @guvectorize([void(float32[:], float32[:])], '(x)->(x)', target='cuda')
    def foo(inp, out):
        pass
    inp = out = np.zeros(10, dtype=np.float32)
    with self.assertRaises(ValueError) as raises:
        foo(inp, out, out=out)
    msg = "cannot specify argument 'out' as both positional and keyword"
    self.assertEqual(str(raises.exception), msg)