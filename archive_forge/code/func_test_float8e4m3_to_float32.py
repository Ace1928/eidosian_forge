import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def test_float8e4m3_to_float32(self):
    self.assertEqual(numpy_helper.float8e4m3_to_float32(int('1111110', 2)), 448)
    self.assertEqual(numpy_helper.float8e4m3_to_float32(int('1000', 2)), 2 ** (-6))
    self.assertEqual(numpy_helper.float8e4m3_to_float32(int('1', 2)), 2 ** (-9))
    self.assertEqual(numpy_helper.float8e4m3_to_float32(int('111', 2)), 0.875 * 2 ** (-6))
    for f in [0, 1, -1, 0.5, -0.5, 0.1015625, -0.1015625, 2, 3, -2, -3, 448, 2 ** (-6), 2 ** (-9), 0.875 * 2 ** (-6), np.nan]:
        with self.subTest(f=f):
            f32 = np.float32(f)
            f8 = helper.float32_to_float8e4m3(f32)
            assert isinstance(f8, int)
            f32_1 = numpy_helper.float8e4m3_to_float32(np.array([f8]))[0]
            f32_2 = float8e4m3_to_float32(f8)
            if np.isnan(f32):
                assert np.isnan(f32_1)
                assert np.isnan(f32_2)
            else:
                self.assertEqual(f32, f32_1)
                self.assertEqual(f32, f32_2)