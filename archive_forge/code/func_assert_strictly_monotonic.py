import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import uniform as uniform_lib
def assert_strictly_monotonic(array):
    if array[0] < array[-1]:
        assert_strictly_increasing(array)
    else:
        assert_strictly_decreasing(array)