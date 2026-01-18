import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Mean(self) -> None:
    self._test_op_upgrade('Mean', 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]], attrs={'consumed_inputs': [0]})