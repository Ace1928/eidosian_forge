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
def scale_trans(input_tensor, scale=scale, translation=translation, kernel_type=kernel_type, antialias=antialias):
    return image_ops.scale_and_translate(input_tensor, out_shape[1:3], scale=constant_op.constant(scale), translation=constant_op.constant(translation), kernel_type=kernel_type, antialias=antialias)