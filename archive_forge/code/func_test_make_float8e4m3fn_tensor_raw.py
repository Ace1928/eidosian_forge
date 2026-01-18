import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
def test_make_float8e4m3fn_tensor_raw(self) -> None:
    expected = np.array([0, 0.5, 1, 448, 10], dtype=np.float32)
    f8 = np.array([helper.float32_to_float8e4m3(x) for x in expected], dtype=np.uint8)
    packed_values = f8.tobytes()
    y = helper.make_tensor(name='test', data_type=TensorProto.FLOAT8E4M3FN, dims=list(expected.shape), vals=packed_values, raw=True)
    ynp = numpy_helper.to_array(y)
    np.testing.assert_equal(expected, ynp)