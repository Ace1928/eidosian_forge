import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_MaxPool_2(self) -> None:
    self._test_op_upgrade('MaxPool', 8, [[1, 1, 5, 5]], [[1, 1, 4, 4], [1, 1, 4, 4]], output_types=[TensorProto.FLOAT, TensorProto.INT64], attrs={'kernel_shape': [2, 2]})