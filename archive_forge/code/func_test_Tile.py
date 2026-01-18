import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Tile(self) -> None:
    repeats = helper.make_tensor('b', TensorProto.INT64, dims=[3], vals=np.array([1, 2, 3]))
    self._test_op_upgrade('Tile', 6, [[3, 4, 5], [3]], [[3, 8, 15]], [TensorProto.FLOAT, TensorProto.INT64], initializer=[repeats])