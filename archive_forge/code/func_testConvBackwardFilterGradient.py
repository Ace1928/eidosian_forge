import collections
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
@test_util.run_cuda_only
def testConvBackwardFilterGradient(self, rate=1):
    in_shape = LayerShapeNHWC(batch=8, height=64, width=64, channels=8)
    filter_shape = FilterShape2D(height=3, width=3, in_channels=8, out_channels=8)
    in_op = self._random_data_op(in_shape)
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    dilations = [1, rate, rate, 1]
    out_op = self._random_out_op(in_shape, filter_shape, strides, padding, dilations)
    self._assert_reproducible(lambda: nn_ops.conv2d_backprop_filter(in_op, filter_shape, out_op, strides=strides, padding=padding, dilations=dilations))