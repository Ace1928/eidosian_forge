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
def test3D(self):
    labels = np.array([[[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]], [[0.0, 0.5, 0.5, 0.0], [0.5, 0.5, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]).astype(np.float32)
    logits = np.array([[[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]], [[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]], [[5.0, 4.0, 3.0, 2.0], [1.0, 2.0, 3.0, 4.0]]]).astype(np.float32)
    self._testXentND(labels, logits, dim=0)
    self._testXentND(labels, logits, dim=1)
    self._testXentND(labels, logits, dim=-1)