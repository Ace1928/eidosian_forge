import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def test_float8_e5m2fnuz_inf(self):
    x = np.float32(np.inf)
    to = helper.float32_to_float8e5m2(x, fn=True, uz=True)
    back = numpy_helper.float8e5m2_to_float32(to, fn=True, uz=True)
    self.assertEqual(back, 57344)
    x = np.float32(np.inf)
    to = helper.float32_to_float8e5m2(x, fn=True, uz=True, saturate=False)
    back = numpy_helper.float8e5m2_to_float32(to, fn=True, uz=True)
    self.assertTrue(np.isnan(back))
    x = np.float32(-np.inf)
    to = helper.float32_to_float8e5m2(x, fn=True, uz=True)
    back = numpy_helper.float8e5m2_to_float32(to, fn=True, uz=True)
    self.assertEqual(back, -57344)
    x = np.float32(-np.inf)
    to = helper.float32_to_float8e5m2(x, fn=True, uz=True, saturate=False)
    back = numpy_helper.float8e5m2_to_float32(to, fn=True, uz=True)
    self.assertTrue(np.isnan(back))