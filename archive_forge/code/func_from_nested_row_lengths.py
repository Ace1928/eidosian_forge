import functools
import operator
import typing
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@classmethod
@dispatch.add_dispatch_support
def from_nested_row_lengths(cls, flat_values, nested_row_lengths, name=None, validate=True):
    """Creates a `RaggedTensor` from a nested list of `row_lengths` tensors.

    Equivalent to:

    ```python
    result = flat_values
    for row_lengths in reversed(nested_row_lengths):
      result = from_row_lengths(result, row_lengths)
    ```

    Args:
      flat_values: A potentially ragged tensor.
      nested_row_lengths: A list of 1-D integer tensors.  The `i`th tensor is
        used as the `row_lengths` for the `i`th ragged dimension.
      name: A name prefix for the RaggedTensor (optional).
      validate: If true, then use assertions to check that the arguments form
        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,
          since they must be checked for each tensor value.

    Returns:
      A `RaggedTensor` (or `flat_values` if `nested_row_lengths` is empty).
    """
    if not isinstance(validate, bool):
        raise TypeError(f'Argument `validate` must have type bool. Received {validate}.')
    if isinstance(nested_row_lengths, tensor_lib.Tensor):
        raise TypeError(f'Argument `nested_row_lengths` must be a list of Tensors. Received {nested_row_lengths}.')
    with ops.name_scope(name, 'RaggedFromNestedRowlengths', [flat_values] + list(nested_row_lengths)):
        result = flat_values
        for lengths in reversed(nested_row_lengths):
            result = cls.from_row_lengths(result, lengths, validate=validate)
        return result