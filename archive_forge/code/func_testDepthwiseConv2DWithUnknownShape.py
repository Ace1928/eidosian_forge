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
def testDepthwiseConv2DWithUnknownShape(self):
    if not test.is_gpu_available():
        return
    with self.session():
        x = array_ops.placeholder(dtypes.float32)
        f = np.ones([1, 1, 1, 1], np.float32)
        v = nn_impl.depthwise_conv2d(x, f, [1, 1, 1, 1], 'VALID', rate=[2, 1], data_format='NCHW')
        self.assertAllEqual(np.ones([1, 1, 1, 1], np.float32), v.eval(feed_dict={x: np.ones([1, 1, 1, 1], np.float32)}))