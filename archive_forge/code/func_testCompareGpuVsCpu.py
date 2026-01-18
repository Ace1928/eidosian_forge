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
@parameterized.parameters({'batch_size': 1, 'channel_count': 1}, {'batch_size': 2, 'channel_count': 3}, {'batch_size': 5, 'channel_count': 4})
def testCompareGpuVsCpu(self, batch_size, channel_count):
    smaller_shape = [batch_size, 4, 6, channel_count]
    larger_shape = [batch_size, 8, 16, channel_count]
    for params in self._itGen(smaller_shape, larger_shape):
        self._gpuVsCpuCase(*params, dtype=np.float32)