import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Squeeze(self) -> None:
    self._test_op_upgrade('Squeeze', 1, [[2, 1, 3, 4, 1]], [[2, 3, 4]])