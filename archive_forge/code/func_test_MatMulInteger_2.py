import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_MatMulInteger_2(self) -> None:
    self._test_op_upgrade('MatMulInteger', 10, [[2, 3], [3, 4], [], []], [[2, 4]], [TensorProto.INT8, TensorProto.INT8, TensorProto.INT8, TensorProto.INT8], [TensorProto.INT32])