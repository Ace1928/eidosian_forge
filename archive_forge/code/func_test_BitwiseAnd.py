import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_BitwiseAnd(self) -> None:
    self._test_op_upgrade('BitwiseAnd', 18, [[2, 3], [2, 3]], [[2, 3]], [TensorProto.INT16, TensorProto.INT16], [TensorProto.INT16])