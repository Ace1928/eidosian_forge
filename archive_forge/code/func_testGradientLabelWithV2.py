import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.nn_grad import _SoftmaxCrossEntropyWithLogitsGrad  # pylint: disable=unused-import
from tensorflow.python.platform import test
@test_util.run_deprecated_v1
def testGradientLabelWithV2(self):
    with self.cached_session():
        labels = constant_op.constant([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5], shape=[3, 4], dtype=dtypes.float64, name='labels')
        logits = constant_op.constant([0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4], shape=[3, 4], dtype=dtypes.float64, name='logits')
        x = nn_ops.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, name='xent')
        err = gradient_checker.compute_gradient_error(labels, [3, 4], x, [3])
    self.assertLess(err, 5e-08)