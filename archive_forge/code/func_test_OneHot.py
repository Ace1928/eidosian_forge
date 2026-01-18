import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_OneHot(self) -> None:
    self._test_op_upgrade('OneHot', 9, [[3, 4, 5], [], [2]], [[3, 4, 5, 6]])