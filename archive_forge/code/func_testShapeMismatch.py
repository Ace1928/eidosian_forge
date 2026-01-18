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
def testShapeMismatch(self):
    with self.assertRaisesRegex(ValueError, '`labels.shape.rank` must equal `logits.shape.rank - 1`'):
        nn_ops.sparse_softmax_cross_entropy_with_logits_v2(labels=[[0, 2]], logits=[[0.0, 1.0], [2.0, 3.0], [2.0, 3.0]])