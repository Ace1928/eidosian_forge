import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_TopK(self) -> None:
    self._test_op_upgrade('TopK', 1, [[3, 4, 5]], [[3, 4, 2], [3, 4, 2]], output_types=[TensorProto.FLOAT, TensorProto.INT64], attrs={'k': 2})