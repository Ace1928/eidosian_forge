import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Pad(self) -> None:
    self._test_op_upgrade('Pad', 2, [[3, 4]], [[5, 8]], attrs={'pads': [1, 2, 1, 2], 'value': 1.5})