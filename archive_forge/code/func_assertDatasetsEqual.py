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
def assertDatasetsEqual(self, dataset1, dataset2):
    """Checks that datasets are equal. Supports both graph and eager mode."""
    self.assertTrue(structure.are_compatible(dataset_ops.get_structure(dataset1), dataset_ops.get_structure(dataset2)))
    flattened_types = nest.flatten(dataset_ops.get_legacy_output_types(dataset1))
    next1 = self.getNext(dataset1)
    next2 = self.getNext(dataset2)
    while True:
        try:
            op1 = self.evaluate(next1())
        except errors.OutOfRangeError:
            with self.assertRaises(errors.OutOfRangeError):
                self.evaluate(next2())
            break
        op2 = self.evaluate(next2())
        op1 = nest.flatten(op1)
        op2 = nest.flatten(op2)
        assert len(op1) == len(op2)
        for i in range(len(op1)):
            if sparse_tensor.is_sparse(op1[i]) or ragged_tensor.is_ragged(op1[i]):
                self.assertValuesEqual(op1[i], op2[i])
            elif flattened_types[i] == dtypes.string:
                self.assertAllEqual(op1[i], op2[i])
            else:
                self.assertAllClose(op1[i], op2[i])