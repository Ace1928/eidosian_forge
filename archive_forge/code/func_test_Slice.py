import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Slice(self) -> None:
    self._test_op_upgrade('Slice', 1, [[3, 4, 5]], [[3, 2, 2]], attrs={'axes': [1, 2], 'starts': [0, 1], 'ends': [2, 3]})