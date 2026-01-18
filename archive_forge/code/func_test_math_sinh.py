import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def test_math_sinh(self):
    self.unary_template_float32(math_sinh, np.sinh)
    self.unary_template_float64(math_sinh, np.sinh)
    self.unary_template_int64(math_sinh, np.sinh)
    self.unary_template_uint64(math_sinh, np.sinh)