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
def test_make_float8e4m3fn_tensor(self) -> None:
    y = helper.make_tensor('zero_point', TensorProto.FLOAT8E4M3FN, [5], [0, 0.5, 1, 50000, 10.1])
    ynp = numpy_helper.to_array(y)
    expected = np.array([0, 0.5, 1, 448, 10], dtype=np.float32)
    np.testing.assert_equal(expected, ynp)