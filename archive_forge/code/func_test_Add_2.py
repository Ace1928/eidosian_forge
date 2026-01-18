import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Add_2(self) -> None:
    self._test_op_upgrade('Add', 1, [[3, 4, 5], [5]], attrs={'consumed_inputs': [0], 'broadcast': 1})