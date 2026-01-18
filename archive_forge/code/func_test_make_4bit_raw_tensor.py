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
@parameterized.parameterized.expand(itertools.product((TensorProto.UINT4, TensorProto.INT4), ((5, 4, 6), (4, 6, 5), (3, 3), (1,))))
def test_make_4bit_raw_tensor(self, dtype, dims) -> None:
    type_range = {TensorProto.UINT4: (0, 15), TensorProto.INT4: (-8, 7)}
    data = np.random.randint(type_range[dtype][0], high=type_range[dtype][1] + 1, size=dims)
    packed_data = helper.pack_float32_to_4bit(data, signed=dtype == TensorProto.INT4)
    y = helper.make_tensor('packed_int4', dtype, dims, packed_data.tobytes(), raw=True)
    ynp = numpy_helper.to_array(y)
    np.testing.assert_equal(data, ynp)