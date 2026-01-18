import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_ArgMin_2(self) -> None:
    self._test_op_upgrade('ArgMin', 7, [[2, 3, 4]], [[2, 1, 4]], output_types=[TensorProto.INT64], attrs={'axis': 1})