import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
@skip_on_cudasim('trunc only supported on NumPy float64')
def test_math_trunc_non_float64(self):
    self.unary_template_float32(math_trunc, np.trunc)
    self.unary_template_int64(math_trunc, np.trunc)
    self.unary_template_uint64(math_trunc, np.trunc)