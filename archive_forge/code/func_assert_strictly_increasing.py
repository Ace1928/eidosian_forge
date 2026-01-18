import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import uniform as uniform_lib
def assert_strictly_increasing(array):
    np.testing.assert_array_less(0.0, np.diff(array))