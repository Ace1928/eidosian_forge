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
def testGradFromResizeToLargerInBothDims(self):
    in_shape = [1, 2, 3, 1]
    out_shape = [1, 4, 6, 1]
    x = np.arange(0, 6).reshape(in_shape).astype(np.float32)
    input_tensor = constant_op.constant(x, shape=in_shape)
    for align_corners in [True, False]:

        def func(input_tensor, align_corners=align_corners):
            return image_ops.resize_bicubic(input_tensor, out_shape[1:3], align_corners=align_corners)
        with self.cached_session():
            err = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(func, [input_tensor]))
        self.assertLess(err, 0.001)