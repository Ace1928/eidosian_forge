import os
import random
import re
from tensorflow.python.data.experimental.ops import lookup_ops as data_lookup_ops
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import test_mode
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
def assertValuesEqual(self, expected, actual):
    """Asserts that two values are equal."""
    if isinstance(expected, dict):
        self.assertItemsEqual(list(expected.keys()), list(actual.keys()))
        for k in expected.keys():
            self.assertValuesEqual(expected[k], actual[k])
    elif sparse_tensor.is_sparse(expected):
        self.assertAllEqual(expected.indices, actual.indices)
        self.assertAllEqual(expected.values, actual.values)
        self.assertAllEqual(expected.dense_shape, actual.dense_shape)
    else:
        self.assertAllEqual(expected, actual)