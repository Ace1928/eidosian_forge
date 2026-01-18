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
@pytest.mark.parametrize('tensor_dtype', [t for t in helper.get_all_tensor_dtypes() if t not in {TensorProto.BFLOAT16, TensorProto.STRING, TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT8E4M3FNUZ, TensorProto.FLOAT8E5M2, TensorProto.FLOAT8E5M2FNUZ, TensorProto.UINT4, TensorProto.INT4}], ids=lambda tensor_dtype: helper.tensor_dtype_to_string(tensor_dtype))
def test_make_tensor_raw(tensor_dtype: int) -> None:
    np_array = np.random.randn(2, 3).astype(helper.tensor_dtype_to_np_dtype(tensor_dtype))
    tensor = helper.make_tensor(name='test', data_type=tensor_dtype, dims=np_array.shape, vals=np_array.tobytes(), raw=True)
    np.testing.assert_equal(np_array, numpy_helper.to_array(tensor))