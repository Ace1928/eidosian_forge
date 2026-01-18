import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_ReduceSum(self) -> None:
    self._test_op_upgrade('ReduceSum', 1, [[3, 4, 5]], [[1, 1, 1]])