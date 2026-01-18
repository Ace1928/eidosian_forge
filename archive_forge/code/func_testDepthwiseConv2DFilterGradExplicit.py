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
def testDepthwiseConv2DFilterGradExplicit(self):
    for index, (input_size, filter_size, output_size, stride, padding, dilations) in enumerate(CheckGradConfigsToTestExplicit()):
        tf_logging.info('Testing DepthwiseConv2DFilterGradExplicit, %dth config: %r * %r, stride: %d, padding: %s', index, input_size, filter_size, stride, padding)
        data_types = [dtypes.float16, dtypes.float32]
        if not test.is_built_with_rocm():
            data_types.extend([dtypes.float64, dtypes.bfloat16])
        data_formats = ['NHWC', 'NCHW'] if test.is_gpu_available() else ['NHWC']
        for data_type in data_types:
            for data_format in data_formats:
                self._ConstructAndTestGradient(input_size, filter_size, output_size, stride, padding, data_type, test_input=False, use_gpu=True, data_format=data_format, dilations=dilations)