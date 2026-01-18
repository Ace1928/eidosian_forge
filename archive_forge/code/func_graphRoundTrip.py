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
def graphRoundTrip(self, dataset, allow_stateful=False):
    """Converts a dataset to a graph and back."""
    graph = gen_dataset_ops.dataset_to_graph(dataset._variant_tensor, allow_stateful=allow_stateful)
    return dataset_ops.from_variant(gen_experimental_dataset_ops.dataset_from_graph(graph), dataset.element_spec)