import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_QLinearMatMul(self) -> None:
    self._test_op_upgrade('QLinearMatMul', 10, [[2, 3], [], [], [3, 4], [], [], [], []], [[2, 4]])