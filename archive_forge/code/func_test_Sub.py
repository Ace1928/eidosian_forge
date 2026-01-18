import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Sub(self) -> None:
    self._test_op_upgrade('Sub', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]], attrs={'consumed_inputs': [0]})