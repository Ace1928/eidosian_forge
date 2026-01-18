import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_GRU_2(self) -> None:
    self._test_op_upgrade('GRU', 7, [[5, 3, 4], [2, 18, 4], [2, 18, 4]], [[5, 2, 3, 6], [2, 3, 6]], attrs={'hidden_size': 6, 'direction': 'bidirectional'})