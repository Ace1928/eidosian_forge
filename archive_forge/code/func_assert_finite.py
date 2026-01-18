import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import uniform as uniform_lib
def assert_finite(array):
    if not np.isfinite(array).all():
        raise AssertionError('array was not all finite. %s' % array[:15])