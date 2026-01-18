import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_string_tensor(self) -> None:
    tensor = TensorProto()
    tensor.data_type = TensorProto.STRING
    tensor.dims.append(1)
    tensor.string_data.append(b'Test')
    checker.check_tensor(tensor)
    del tensor.string_data[:]
    tensor.raw_data = b'Test'
    self.assertRaises(checker.ValidationError, checker.check_tensor, tensor)