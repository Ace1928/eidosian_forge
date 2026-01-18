import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
@parameterized.parameterized.expand([(0.00439453125, 0.00390625), (0.005859375, 0.005859375), (0.005759375, 0.005859375), (0.0046875, 0.00390625), (0.001953125, 0.001953125), (0.0029296875, 0.00390625), (0.002053125, 0.001953125), (0.00234375, 0.001953125), (0.0087890625, 0.0078125), (0.001171875, 0.001953125), (1.8131605, 1.875)])
def test_float8e4m3_to_float32_round(self, val, expected):
    f8 = helper.float32_to_float8e4m3(val)
    f32 = numpy_helper.float8e4m3_to_float32(f8)
    self.assertEqual(f32, expected)