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
@test_util.run_v1_only('b/120545219')
@test_util.run_cuda_only
def testDepthwiseConv2DCudnn(self):
    for index, (input_size, filter_size, _, stride, padding, dilations) in enumerate(ConfigsToTest()):
        tf_logging.info('Testing DepthwiseConv2DCudnn, %dth config: %r * %r, stride: %d, padding: %s', index, input_size, filter_size, stride, padding)
        data_types = [dtypes.float16, dtypes.bfloat16]
        for data_type in data_types:
            self._VerifyValues(input_size, filter_size, stride, padding, data_type, use_gpu=True, data_format='NCHW', dilations=dilations)