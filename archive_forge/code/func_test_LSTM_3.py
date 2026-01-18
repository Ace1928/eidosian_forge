import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_LSTM_3(self) -> None:
    self._test_op_upgrade('LSTM', 7, [[5, 3, 4], [1, 24, 4], [1, 24, 4], [1, 48], [5], [1, 5, 6], [1, 5, 6], [1, 18]], [[5, 1, 3, 6], [1, 3, 6], [1, 3, 6]], [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT], attrs={'hidden_size': 6})