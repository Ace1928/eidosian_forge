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
def testLabelsBroadcast(self):
    """This method is structured to be easily overridden by a child class."""
    self._testLabelsBroadcast(uniform_labels_gradient=[[0.25, 0.25, 0.25, 0.25], [-1.968, -1.913, -1.763, -1.355], [-0.218, -0.163, -0.013, 0.394]])