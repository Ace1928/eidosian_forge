import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_GRU_1(self) -> None:
    self._test_op_upgrade('GRU', 7, [[5, 3, 4], [1, 18, 4], [1, 18, 4]], [[5, 1, 3, 6], [1, 3, 6]], attrs={'hidden_size': 6})