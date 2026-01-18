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
def from_nested_value_rowids(cls, flat_values, nested_value_rowids, nested_nrows=None, name=None, validate=True):
    """Creates a `RaggedTensor` from a nested list of `value_rowids` tensors.

    Equivalent to:

    ```python
    result = flat_values
    for (rowids, nrows) in reversed(zip(nested_value_rowids, nested_nrows)):
      result = from_value_rowids(result, rowids, nrows)
    ```

    Args:
      flat_values: A potentially ragged tensor.
      nested_value_rowids: A list of 1-D integer tensors.  The `i`th tensor is
        used as the `value_rowids` for the `i`th ragged dimension.
      nested_nrows: A list of integer scalars.  The `i`th scalar is used as the
        `nrows` for the `i`th ragged dimension.
      name: A name prefix for the RaggedTensor (optional).
      validate: If true, then use assertions to check that the arguments form
        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,
          since they must be checked for each tensor value.

    Returns:
      A `RaggedTensor` (or `flat_values` if `nested_value_rowids` is empty).

    Raises:
      ValueError: If `len(nested_values_rowids) != len(nested_nrows)`.
    """
    if not isinstance(validate, bool):
        raise TypeError(f'Argument `validate` must have type bool. Received {validate}.')
    if isinstance(nested_value_rowids, tensor_lib.Tensor):
        raise TypeError(f'Argument `nested_value_rowids` must be a list of Tensors. Received {nested_value_rowids}.')
    if nested_nrows is None:
        nested_nrows = [None] * len(nested_value_rowids)
    else:
        if isinstance(nested_nrows, tensor_lib.Tensor):
            raise TypeError(f'Argument `nested_nrows` must be a list of Tensors. Received {nested_nrows}.')
        if len(nested_nrows) != len(nested_value_rowids):
            raise ValueError(f'Argument `nested_nrows` must have the same length as argument `nested_value_rowids`. len(nested_nrows) = {len(nested_nrows)} vs. len(nested_values_rowids) = {len(nested_value_rowids)}.')
    with ops.name_scope(name, 'RaggedFromNestedValueRowIds', [flat_values] + list(nested_value_rowids) + list(nested_nrows)):
        result = flat_values
        for value_rowids, nrows in reversed(list(zip(nested_value_rowids, nested_nrows))):
            result = cls.from_value_rowids(result, value_rowids, nrows, validate=validate)
        return result