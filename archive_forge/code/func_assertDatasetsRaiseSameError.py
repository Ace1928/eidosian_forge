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
def assertDatasetsRaiseSameError(self, dataset1, dataset2, exception_class, replacements=None):
    """Checks that datasets raise the same error on the first get_next call."""
    if replacements is None:
        replacements = []
    next1 = self.getNext(dataset1)
    next2 = self.getNext(dataset2)
    try:
        self.evaluate(next1())
        raise ValueError('Expected dataset to raise an error of type %s, but it did not.' % repr(exception_class))
    except exception_class as e:
        expected_message = e.message
        for old, new, count in replacements:
            expected_message = expected_message.replace(old, new, count)
        with self.assertRaisesRegexp(exception_class, re.escape(expected_message)):
            self.evaluate(next2())