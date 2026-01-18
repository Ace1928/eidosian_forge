import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
@test_util.run_gpu_only
def testDepthwiseConv2DForwardCompare(self):
    for index, (input_size, filter_size, output_size, stride, padding, dilations) in enumerate(ConfigsToTest()):
        if dilations:
            continue
        tf_logging.info('Testing DepthwiseConv2DForwardCompare, %dth config: %r * %r, stride: %d, padding: %s', index, input_size, filter_size, stride, padding)
        self._CompareForward(input_size, filter_size, output_size, stride, padding, 'float32')
        if test.is_built_with_rocm():
            continue
        self._CompareForward(input_size, filter_size, output_size, stride, padding, 'float64')
        self._CompareForward(input_size, filter_size, output_size, stride, padding, 'bfloat16')