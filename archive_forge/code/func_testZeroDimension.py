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
def testZeroDimension(self):
    labels = np.zeros([0, 2, 4]).astype(np.float32)
    logits = np.zeros([0, 2, 4]).astype(np.float32)
    np_loss, _ = self._npXent(labels=labels, logits=logits)
    loss = nn_ops.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    tf_loss = self.evaluate(loss)
    self.assertAllEqual(np_loss, tf_loss)