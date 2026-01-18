import numpy as np
from tensorflow.python.eager import backprop as backprop_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
def testNpXent(self):
    labels = [3, 0]
    logits = [[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]
    np_loss, np_gradient = self._npXent(labels=np.array(labels), logits=np.array(logits))
    self.assertAllClose(np.array([[0.25, 0.25, 0.25, -0.75], [-0.968, 0.087, 0.237, 0.6439]]), np_gradient, rtol=0.001, atol=0.001)
    self.assertAllClose(np.array([1.3862, 3.442]), np_loss, rtol=0.001, atol=0.001)