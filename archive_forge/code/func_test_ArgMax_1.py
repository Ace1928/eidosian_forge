import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_ArgMax_1(self) -> None:
    self._test_op_upgrade('ArgMax', 7, [[2, 3, 4]], [[1, 3, 4]], output_types=[TensorProto.INT64])