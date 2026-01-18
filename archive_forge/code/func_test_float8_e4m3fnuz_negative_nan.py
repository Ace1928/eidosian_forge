import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def test_float8_e4m3fnuz_negative_nan(self):
    x = numpy_helper.float8e5m2_to_float32(255)
    to = helper.float32_to_float8e4m3(x, uz=True)
    self.assertEqual(to, 128)
    back = numpy_helper.float8e4m3_to_float32(to, uz=True)
    self.assertTrue(np.isnan(back))
    x = numpy_helper.float8e5m2_to_float32(255)
    to = helper.float32_to_float8e4m3(x, uz=True, saturate=False)
    self.assertEqual(to, 128)
    back = numpy_helper.float8e4m3_to_float32(to, uz=True)
    self.assertTrue(np.isnan(back))