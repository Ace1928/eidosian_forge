import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Einsum_2(self) -> None:
    self._test_op_upgrade('Einsum', 12, [[4, 5]], [[5, 4]], attrs={'equation': 'ij->ji'})