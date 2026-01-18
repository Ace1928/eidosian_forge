import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@classmethod
def from_pyval(cls, pyval, typespec=None):
    """Constructs a StructuredTensor from a nested Python structure.

    >>> tf.experimental.StructuredTensor.from_pyval(
    ...     {'a': [1, 2, 3], 'b': [[4, 5], [6, 7]]})
    <StructuredTensor(
        fields={
          "a": tf.Tensor([1 2 3], shape=(3,), dtype=int32),
          "b": <tf.RaggedTensor [[4, 5], [6, 7]]>},
        shape=())>

    Note that `StructuredTensor.from_pyval(pyval).to_pyval() == pyval`.

    Args:
      pyval: The nested Python structure that should be used to create the new
        `StructuredTensor`.
      typespec: A `StructuredTensor.Spec` specifying the expected type for each
        field. If not specified, then all nested dictionaries are turned into
        StructuredTensors, and all nested lists are turned into Tensors (if
        rank<2) or RaggedTensors (if rank>=2).

    Returns:
      A `StructuredTensor`.
    """
    return cls._from_pyval(pyval, typespec, ())