import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_tensor(self) -> None:
    tensor = self._sample_float_tensor
    checker.check_tensor(tensor)
    tensor.raw_data = np.random.randn(2, 3).astype(np.float32).tobytes()
    self.assertRaises(checker.ValidationError, checker.check_tensor, tensor)