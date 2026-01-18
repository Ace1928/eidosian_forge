import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_RNN_2(self) -> None:
    self._test_op_upgrade('RNN', 7, [[5, 3, 4], [2, 6, 4], [2, 6, 4]], [[5, 2, 3, 6], [2, 3, 6]], attrs={'hidden_size': 6, 'direction': 'bidirectional'})