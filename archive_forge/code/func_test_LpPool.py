import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_LpPool(self) -> None:
    self._test_op_upgrade('LpPool', 2, [[1, 1, 5, 5]], [[1, 1, 4, 4]], attrs={'kernel_shape': [2, 2]})