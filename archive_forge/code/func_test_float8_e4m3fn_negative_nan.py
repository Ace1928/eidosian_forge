import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def test_float8_e4m3fn_negative_nan(self):
    x = numpy_helper.float8e5m2_to_float32(255)
    to = helper.float32_to_float8e4m3(x)
    self.assertEqual(to, 255)
    back = numpy_helper.float8e4m3_to_float32(to)
    self.assertTrue(np.isnan(back))
    x = numpy_helper.float8e5m2_to_float32(255)
    to = helper.float32_to_float8e4m3(x, saturate=False)
    self.assertEqual(to, 255)
    back = numpy_helper.float8e4m3_to_float32(to)
    self.assertTrue(np.isnan(back))