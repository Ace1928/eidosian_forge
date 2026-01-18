import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_SpaceToDepth(self) -> None:
    self._test_op_upgrade('SpaceToDepth', 1, [[1, 3, 8, 8]], [[1, 12, 4, 4]], attrs={'blocksize': 2})