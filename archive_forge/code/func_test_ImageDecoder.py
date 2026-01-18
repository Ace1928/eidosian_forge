import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_ImageDecoder(self) -> None:
    self._test_op_upgrade('ImageDecoder', 20, [[None]], [[None, None, 3]], input_types=[TensorProto.UINT8], output_types=[TensorProto.UINT8])