import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Where(self) -> None:
    self._test_op_upgrade('Where', 9, [[2, 3], [2, 3], [2, 3]], [[2, 3]], [TensorProto.BOOL, TensorProto.FLOAT, TensorProto.FLOAT])