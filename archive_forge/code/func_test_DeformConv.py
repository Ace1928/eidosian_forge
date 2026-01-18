import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_DeformConv(self) -> None:
    self._test_op_upgrade('DeformConv', 19, [[1, 1, 3, 3], [1, 1, 2, 2], [1, 8, 2, 2]], [[1, 1, 2, 2]])