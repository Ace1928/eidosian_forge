import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Div(self) -> None:
    self._test_op_upgrade('Div', 1, [[3, 4, 5], [3, 1, 5]], attrs={'consumed_inputs': [0]})