from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
def f_dummy(x):
    r = x[..., 0]
    g = x[..., 1]
    b = x[..., 2]
    v = r
    s = 1 - math_ops.div_no_nan(b, r)
    h = 60 * math_ops.div_no_nan(g - b, r - b)
    h = h / 360
    return array_ops_stack.stack([h, s, v], axis=-1)