import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def binary_template_uint64(self, func, npfunc, start=0, stop=50):
    self.binary_template(func, npfunc, np.uint64, np.float64, start, stop)