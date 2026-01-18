import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_BatchNormalization_2(self) -> None:
    self._test_op_upgrade('BatchNormalization', 14, [[1, 3], [3], [3], [3], [3]], [[1, 3], [3], [3]], attrs={'training_mode': 1})