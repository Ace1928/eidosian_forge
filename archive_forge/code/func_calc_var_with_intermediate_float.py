import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def calc_var_with_intermediate_float(input):
    vals_c = input - input.mean()
    count = vals_c.size
    return cupy.square(vals_c).sum() / cupy.asanyarray(count).astype(float)