import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Selu(self) -> None:
    self._test_op_upgrade('Selu', 1, attrs={'consumed_inputs': [0]})