import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Upsample(self) -> None:
    self._test_op_upgrade('Upsample', 1, [[1, 3, 4, 5]], [[1, 3, 6, 10]], attrs={'width_scale': 2.0, 'height_scale': 1.5})