import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def test_int8(self) -> None:
    self._test_numpy_helper_int_type(np.int8)