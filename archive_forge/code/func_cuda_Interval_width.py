from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import numpy as np
from numba import config, cuda, njit, types
@cuda_lower_attr(IntervalType, 'width')
def cuda_Interval_width(context, builder, sig, arg):
    lo = builder.extract_value(arg, 0)
    hi = builder.extract_value(arg, 1)
    return builder.fsub(hi, lo)