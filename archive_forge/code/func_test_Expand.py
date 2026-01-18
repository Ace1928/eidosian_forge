import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Expand(self) -> None:
    shape = helper.make_tensor('b', TensorProto.INT64, dims=[4], vals=np.array([5, 2, 6, 4]))
    self._test_op_upgrade('Expand', 8, [[2, 1, 4], [4]], [[5, 2, 6, 4]], [TensorProto.FLOAT, TensorProto.INT64], initializer=[shape])