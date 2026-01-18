import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Unsqueeze(self) -> None:
    self._test_op_upgrade('Unsqueeze', 1, [[3, 4, 5]], [[3, 4, 1, 5]], attrs={'axes': [2]})