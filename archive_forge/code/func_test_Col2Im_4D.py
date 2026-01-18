import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Col2Im_4D(self) -> None:
    self._test_op_upgrade('Col2Im', 18, [[1, 5, 5], [2], [2]], [[1, 1, 5, 5]])