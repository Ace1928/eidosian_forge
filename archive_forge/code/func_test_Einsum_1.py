import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Einsum_1(self) -> None:
    self._test_op_upgrade('Einsum', 12, [[3, 4, 5], [3, 5, 6]], [[3, 4, 6]], attrs={'equation': 'bij, bjk -> bik'})