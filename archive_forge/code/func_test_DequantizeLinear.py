import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_DequantizeLinear(self) -> None:
    self._test_op_upgrade('DequantizeLinear', 10, [[2, 3], [], []], [[2, 3]], [TensorProto.INT8, TensorProto.FLOAT, TensorProto.INT8])