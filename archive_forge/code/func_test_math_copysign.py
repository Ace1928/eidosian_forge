import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def test_math_copysign(self):
    self.binary_template_float32(math_copysign, np.copysign, start=-1)
    self.binary_template_float64(math_copysign, np.copysign, start=-1)