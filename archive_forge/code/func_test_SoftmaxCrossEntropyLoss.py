import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_SoftmaxCrossEntropyLoss(self) -> None:
    self._test_op_upgrade('SoftmaxCrossEntropyLoss', 12, [[3, 4, 5, 6], [3, 6]], [[]], [TensorProto.FLOAT, TensorProto.INT64])