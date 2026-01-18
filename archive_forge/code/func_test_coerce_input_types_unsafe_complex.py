import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
@skip_on_cudasim('Simulator ignores signature')
def test_coerce_input_types_unsafe_complex(self):
    c_add = cuda.jit('(i4[::1], i4, i4)')(add_kernel)
    r = np.zeros(1, dtype=np.int32)
    with self.assertRaises(TypeError):
        c_add[1, 1](r, 12.3, 45.6j)