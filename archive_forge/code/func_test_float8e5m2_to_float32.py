import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def test_float8e5m2_to_float32(self):
    self.assertEqual(numpy_helper.float8e5m2_to_float32(int('1111011', 2)), 57344)
    self.assertEqual(numpy_helper.float8e5m2_to_float32(int('100', 2)), 2 ** (-14))
    self.assertEqual(numpy_helper.float8e5m2_to_float32(int('11', 2)), 0.75 * 2 ** (-14))
    self.assertEqual(numpy_helper.float8e5m2_to_float32(int('1', 2)), 2 ** (-16))
    self.assertTrue(np.isnan(numpy_helper.float8e5m2_to_float32(int('1111101', 2))))
    self.assertTrue(np.isnan(numpy_helper.float8e5m2_to_float32(int('1111110', 2))))
    self.assertTrue(np.isnan(numpy_helper.float8e5m2_to_float32(int('1111111', 2))))
    self.assertTrue(np.isnan(numpy_helper.float8e5m2_to_float32(int('11111101', 2))))
    self.assertTrue(np.isnan(numpy_helper.float8e5m2_to_float32(int('11111110', 2))))
    self.assertTrue(np.isnan(numpy_helper.float8e5m2_to_float32(int('11111111', 2))))
    self.assertEqual(numpy_helper.float8e5m2_to_float32(int('1111100', 2)), np.inf)
    self.assertEqual(numpy_helper.float8e5m2_to_float32(int('11111100', 2)), -np.inf)
    for f in [0, 0.0017089844, 20480, 14, -3584, np.nan]:
        with self.subTest(f=f):
            f32 = np.float32(f)
            f8 = helper.float32_to_float8e5m2(f32)
            assert isinstance(f8, int)
            f32_1 = numpy_helper.float8e5m2_to_float32(np.array([f8]))[0]
            f32_2 = float8e5m2_to_float32(f8)
            if np.isnan(f32):
                assert np.isnan(f32_1)
                assert np.isnan(f32_2)
            else:
                self.assertEqual(f32, f32_1)
                self.assertEqual(f32, f32_2)