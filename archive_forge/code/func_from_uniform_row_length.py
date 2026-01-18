import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
@classmethod
def from_uniform_row_length(cls, uniform_row_length, nvals=None, nrows=None, validate=True, dtype=None, dtype_hint=None):
    """Creates a `RowPartition` with rows partitioned by `uniform_row_length`.

    This `RowPartition` divides a sequence `values` into rows that all have
    the same length:

    ```python
    partitioned_rows = [[values.pop(0) for _ in range(uniform_row_length)]
             for _ in range(nrows)]
    ```

    Note that either or both of nvals and nrows must be specified.

    Args:
      uniform_row_length: A scalar integer tensor.  Must be nonnegative. The
        size of the outer axis of `values` must be evenly divisible by
        `uniform_row_length`.
      nvals: a non-negative scalar integer tensor for the number of values.
        Must be specified if nrows is not specified. If not specified,
        defaults to uniform_row_length*nrows
      nrows: The number of rows in the constructed RowPartition.  If not
        specified, then it defaults to `nvals/uniform_row_length` (or `0` if
        `uniform_row_length==0`).  `nrows` only needs to be specified if
        `uniform_row_length` might be zero.  `uniform_row_length*nrows` must be
        `nvals`.
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      dtype: Optional dtype for the RowPartition. If missing, the type
        is inferred from the type of `uniform_row_length`, dtype_hint,
        or tf.int64.
      dtype_hint: Optional dtype for the RowPartition, used when dtype
        is None. In some cases, a caller may not have a dtype in mind when
        converting to a tensor, so dtype_hint can be used as a soft preference.
        If the conversion to `dtype_hint` is not possible, this argument has no
        effect.

    Returns:
      A `RowPartition`.
    """
    if not isinstance(validate, bool):
        raise TypeError('validate must have type bool')
    if nrows is None and nvals is None:
        raise ValueError('Either (or both) of nvals and nrows must be specified')
    with ops.name_scope(None, 'RowPartitionFromUniformRowLength', [uniform_row_length, nrows]):
        [uniform_row_length, nvals, nrows] = _convert_all_to_tensors([(uniform_row_length, 'uniform_row_length'), (nvals, 'nvals'), (nrows, 'nrows')], dtype=dtype, dtype_hint=dtype_hint)
        uniform_row_length.shape.assert_has_rank(0)
        const_row_length = tensor_util.constant_value(uniform_row_length)
        if nrows is None:
            if const_row_length is None:
                rowlen_or_1 = math_ops.maximum(uniform_row_length, constant_op.constant(1, uniform_row_length.dtype))
                nrows = nvals // rowlen_or_1
            elif const_row_length == 0:
                nrows = constant_op.constant(0, dtype=uniform_row_length.dtype)
            else:
                nrows = nvals // const_row_length
        const_nrows = None if nrows is None else tensor_util.constant_value(nrows)
        const_nvals = None if nvals is None else tensor_util.constant_value(nvals)
        const_uniform_row_length = tensor_util.constant_value(uniform_row_length)
        checks = []
        if const_nvals is None and const_nrows is not None and (const_uniform_row_length is not None):
            const_nvals = const_nrows * const_uniform_row_length
            if nvals is not None and validate:
                checks.append(check_ops.assert_equal(nvals, const_nvals))
            nvals = constant_op.constant(const_nvals, uniform_row_length.dtype)
        if nvals is None:
            nvals = nrows * uniform_row_length
        if const_nrows is not None and const_row_length is not None:
            row_splits = [v * const_row_length for v in range(const_nrows + 1)]
            row_splits = constant_op.constant(row_splits, uniform_row_length.dtype)
        else:
            row_splits = math_ops.range(nrows + 1, dtype=uniform_row_length.dtype) * uniform_row_length
        if validate:
            if const_nrows is None or const_row_length is None or const_nvals is None:
                checks.append(check_ops.assert_equal(nrows * uniform_row_length, nvals, ('uniform_row_length', uniform_row_length, 'times nrows', nrows, 'must equal nvals', nvals)))
            elif const_nrows * const_row_length != const_nvals:
                raise ValueError('uniform_row_length=%d times nrows=%d must equal nvals=%d' % (const_row_length, const_nrows, const_nvals))
            if uniform_row_length.shape.rank is None:
                checks.append(check_ops.assert_rank(uniform_row_length, 0, message='uniform_row_length must be a scalar.'))
            const_row_length = tensor_util.constant_value(uniform_row_length)
            if const_row_length is None:
                checks.append(check_ops.assert_greater_equal(uniform_row_length, constant_op.constant(0, uniform_row_length.dtype), message='uniform_row_length must be >= 0.'))
            elif const_row_length < 0:
                raise ValueError('uniform_row_length must be >= 0.')
            row_splits = control_flow_ops.with_dependencies(checks, row_splits)
        return cls(row_splits=row_splits, uniform_row_length=uniform_row_length, nrows=nrows, nvals=nvals, internal=_row_partition_factory_key)