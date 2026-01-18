import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
@parameterized.parameterized.expand([(1,), (0.100097656,), (130048,), (1.2993813e-05,), (np.nan,), (np.inf,)])
def test_bfloat16_to_float32(self, f):
    f32 = np.float32(f)
    bf16 = helper.float32_to_bfloat16(f32)
    assert isinstance(bf16, int)
    f32_1 = numpy_helper.bfloat16_to_float32(np.array([bf16]))[0]
    f32_2 = bfloat16_to_float32(bf16)
    if np.isnan(f32):
        assert np.isnan(f32_1)
        assert np.isnan(f32_2)
    else:
        self.assertEqual(f32, f32_1)
        self.assertEqual(f32, f32_2)