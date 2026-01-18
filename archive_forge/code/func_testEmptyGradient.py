import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
def testEmptyGradient(self):
    for data_format, use_gpu in (('NHWC', False), ('NHWC', True)):
        for shape in ((0, 0), (2, 0), (0, 2)):
            self._testGradient(np.random.randn(*shape), np.random.randn(shape[-1]), dtypes.float64, data_format, use_gpu)
    for data_format, use_gpu in [('NHWC', False), ('NHWC', True), ('NCHW', False), ('NCHW', True)]:
        for shape in ((4, 3, 0), (4, 0, 3), (0, 4, 3)):
            self._testGradient(np.random.randn(*shape), np.random.randn(shape[-1]), dtypes.float64, data_format, use_gpu)