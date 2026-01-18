import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_GridSample(self) -> None:
    self._test_op_upgrade('GridSample', 16, [[1, 1, 3, 3], [1, 3, 3, 2]], [[1, 1, 3, 3]], input_types=[TensorProto.FLOAT, TensorProto.FLOAT], output_types=[TensorProto.FLOAT], attrs={'mode': 'nearest', 'padding_mode': 'border', 'align_corners': 1})