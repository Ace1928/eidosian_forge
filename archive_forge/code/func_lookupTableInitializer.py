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
def lookupTableInitializer(self, init_source, vals):
    """Returns a lookup table initializer for the given source and values.

    Args:
      init_source: One of ["textfile", "keyvalue", "dataset"], indicating what
        type of initializer to use.
      vals: The initializer values. The keys will be `range(len(vals))`.
    """
    if init_source == 'textfile':
        return self.textFileInitializer(vals)
    elif init_source == 'keyvaluetensor':
        return self.keyValueTensorInitializer(vals)
    elif init_source == 'dataset':
        return self.datasetInitializer(vals)
    else:
        raise ValueError('Unrecognized init_source: ' + init_source)