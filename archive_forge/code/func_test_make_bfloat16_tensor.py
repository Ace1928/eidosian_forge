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
def test_make_bfloat16_tensor(self) -> None:
    np_array = np.array([[1.0, 2.0], [3.0, 4.0], [0.099853515625, 0.099365234375], [0.0998535081744, 0.1], [np.nan, np.inf]], dtype=np.float32)
    np_results = np.array([[struct.unpack('!f', bytes.fromhex('3F800000'))[0], struct.unpack('!f', bytes.fromhex('40000000'))[0]], [struct.unpack('!f', bytes.fromhex('40400000'))[0], struct.unpack('!f', bytes.fromhex('40800000'))[0]], [struct.unpack('!f', bytes.fromhex('3DCC0000'))[0], struct.unpack('!f', bytes.fromhex('3DCC0000'))[0]], [struct.unpack('!f', bytes.fromhex('3DCC0000'))[0], struct.unpack('!f', bytes.fromhex('3DCD0000'))[0]], [struct.unpack('!f', bytes.fromhex('7FC00000'))[0], struct.unpack('!f', bytes.fromhex('7F800000'))[0]]])
    tensor = helper.make_tensor(name='test', data_type=TensorProto.BFLOAT16, dims=np_array.shape, vals=np_array)
    self.assertEqual(tensor.name, 'test')
    np.testing.assert_equal(np_results, numpy_helper.to_array(tensor))