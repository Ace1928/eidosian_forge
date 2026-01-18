import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Softmax_3(self) -> None:
    self._test_op_upgrade('Softmax', 1, attrs={'axis': -1})