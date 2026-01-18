import os
import numpy as np
from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.data.experimental.ops import iterator_ops as contrib_iterator_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.util import nest
def match(self, expected, actual):
    """Matches nested structures.

    Recursively matches shape and values of `expected` and `actual`.
    Handles scalars, numpy arrays and other python sequence containers
    e.g. list, dict, as well as SparseTensorValue and RaggedTensorValue.

    Args:
      expected: Nested structure 1.
      actual: Nested structure 2.

    Raises:
      AssertionError if matching fails.
    """
    if isinstance(expected, np.ndarray):
        expected = expected.tolist()
    if isinstance(actual, np.ndarray):
        actual = actual.tolist()
    self.assertEqual(type(expected), type(actual))
    if nest.is_nested(expected):
        self.assertEqual(len(expected), len(actual))
        if isinstance(expected, dict):
            for key1, key2 in zip(sorted(expected), sorted(actual)):
                self.assertEqual(key1, key2)
                self.match(expected[key1], actual[key2])
        else:
            for item1, item2 in zip(expected, actual):
                self.match(item1, item2)
    elif isinstance(expected, sparse_tensor.SparseTensorValue):
        self.match((expected.indices, expected.values, expected.dense_shape), (actual.indices, actual.values, actual.dense_shape))
    elif isinstance(expected, ragged_tensor_value.RaggedTensorValue):
        self.match((expected.values, expected.row_splits), (actual.values, actual.row_splits))
    else:
        self.assertEqual(expected, actual)