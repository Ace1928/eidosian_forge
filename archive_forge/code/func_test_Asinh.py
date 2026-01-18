import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Asinh(self) -> None:
    self._test_op_upgrade('Asinh', 9)