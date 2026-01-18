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
def test_attr_sparse_tensor_proto(self) -> None:
    dense_shape = [3, 3]
    sparse_values = [1.764052391052246, 0.40015721321105957, 0.978738009929657]
    values_tensor = helper.make_tensor(name='sparse_values', data_type=TensorProto.FLOAT, dims=[len(sparse_values)], vals=np.array(sparse_values).astype(np.float32), raw=False)
    linear_indices = [2, 3, 5]
    indices_tensor = helper.make_tensor(name='indices', data_type=TensorProto.INT64, dims=[len(linear_indices)], vals=np.array(linear_indices).astype(np.int64), raw=False)
    sparse_tensor = helper.make_sparse_tensor(values_tensor, indices_tensor, dense_shape)
    attr = helper.make_attribute('sparse_attr', sparse_tensor)
    self.assertEqual(attr.name, 'sparse_attr')
    checker.check_sparse_tensor(helper.get_attribute_value(attr))
    checker.check_attribute(attr)