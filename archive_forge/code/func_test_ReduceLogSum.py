import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_ReduceLogSum(self) -> None:
    self._test_op_upgrade('ReduceLogSum', 1, [[3, 4, 5]], [[1, 1, 1]])