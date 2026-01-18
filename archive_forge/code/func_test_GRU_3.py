import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_GRU_3(self) -> None:
    self._test_op_upgrade('GRU', 7, [[5, 3, 4], [1, 18, 4], [1, 18, 4], [1, 24], [5], [1, 5, 6]], [[5, 1, 3, 6], [1, 3, 6]], [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT], attrs={'hidden_size': 6})