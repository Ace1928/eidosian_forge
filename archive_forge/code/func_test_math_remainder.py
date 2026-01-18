import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def test_math_remainder(self):
    self.binary_template_float32(math_remainder, np.remainder, start=1e-11)
    self.binary_template_float64(math_remainder, np.remainder, start=1e-11)
    self.binary_template_int64(math_remainder, np.remainder, start=1)
    self.binary_template_uint64(math_remainder, np.remainder, start=1)