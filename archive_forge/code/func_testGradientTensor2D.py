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
def testGradientTensor2D(self):
    for data_format, use_gpu in (('NHWC', False), ('NHWC', True)):
        for dtype in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16):
            np_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype.as_numpy_dtype).reshape(3, 2)
            bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
            self._testGradient(np_input, bias, dtype, data_format, use_gpu)