import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_RoiAlign_1(self) -> None:
    self._test_op_upgrade('RoiAlign', 10, [[2, 3, 20, 20], [10, 4], [10]], [[10, 3, 1, 1]], [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64])