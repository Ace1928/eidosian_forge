import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Mod_2(self) -> None:
    self._test_op_upgrade('Mod', 10, [[2, 3], [2, 3]], [[2, 3]], attrs={'fmod': 1})