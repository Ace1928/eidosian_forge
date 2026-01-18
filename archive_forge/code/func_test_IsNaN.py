import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_IsNaN(self) -> None:
    self._test_op_upgrade('IsNaN', 9, [[2, 3]], [[2, 3]], output_types=[TensorProto.BOOL])