import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Gather(self) -> None:
    self._test_op_upgrade('Gather', 1, [[3, 4, 5], [6, 7]], [[6, 7, 4, 5]], [TensorProto.FLOAT, TensorProto.INT64])