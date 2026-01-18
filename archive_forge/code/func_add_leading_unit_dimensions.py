import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
def add_leading_unit_dimensions(x, num_dimensions):
    new_shape = array_ops.concat([array_ops.ones([num_dimensions], dtype=dtypes.int32), array_ops.shape(x)], axis=0)
    return array_ops.reshape(x, new_shape)