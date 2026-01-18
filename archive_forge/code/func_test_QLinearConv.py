import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_QLinearConv(self) -> None:
    self._test_op_upgrade('QLinearConv', 10, [[1, 3, 5, 5], [], [], [4, 3, 2, 2], [], [], [], []], [[1, 4, 4, 4]])