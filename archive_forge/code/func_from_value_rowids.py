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
def from_value_rowids(cls, value_rowids, nrows=None, validate=True, dtype=None, dtype_hint=None):
    """Creates a `RowPartition` with rows partitioned by `value_rowids`.

    This `RowPartition` divides a sequence `values` into rows by specifying
    which row each value should be added to:

    ```python
    partitioned_rows = [[] for _ in nrows]
    for (value, rowid) in zip(values, value_rowids):
      partitioned_rows[rowid].append(value)
    ```

    Args:
      value_rowids: A 1-D integer tensor with shape `[nvals]`, which corresponds
        one-to-one with `values`, and specifies each value's row index.  Must be
        nonnegative, and must be sorted in ascending order.
      nrows: An integer scalar specifying the number of rows.  This should be
        specified if the `RowPartition` may containing empty training rows. Must
        be greater than `value_rowids[-1]` (or greater than or equal to zero if
        `value_rowids` is empty). Defaults to `value_rowids[-1] + 1` (or zero if
        `value_rowids` is empty).
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      dtype: Optional dtype for the RowPartition. If missing, the type
        is inferred from the type of `value_rowids`, dtype_hint, or tf.int64.
      dtype_hint: Optional dtype for the RowPartition, used when dtype
        is None. In some cases, a caller may not have a dtype in mind when
        converting to a tensor, so dtype_hint can be used as a soft preference.
        If the conversion to `dtype_hint` is not possible, this argument has no
        effect.

    Returns:
      A `RowPartition`.

    Raises:
      ValueError: If `nrows` is incompatible with `value_rowids`.

    #### Example:

    >>> print(RowPartition.from_value_rowids(
    ...     value_rowids=[0, 0, 0, 0, 2, 2, 2, 3],
    ...     nrows=4))
    tf.RowPartition(row_splits=[0 4 4 7 8])
    """
    from tensorflow.python.ops import bincount_ops
    if not isinstance(validate, bool):
        raise TypeError('validate must have type bool')
    with ops.name_scope(None, 'RowPartitionFromValueRowIds', [value_rowids, nrows]):
        value_rowids = cls._convert_row_partition(value_rowids, 'value_rowids', dtype_hint=dtype_hint, dtype=dtype)
        if nrows is None:
            const_rowids = tensor_util.constant_value(value_rowids)
            if const_rowids is None:
                nrows = array_ops.concat([value_rowids[-1:], [-1]], axis=0)[0] + 1
                const_nrows = None
            else:
                const_nrows = const_rowids[-1] + 1 if const_rowids.size > 0 else 0
                nrows = ops.convert_to_tensor(const_nrows, value_rowids.dtype, name='nrows')
        else:
            nrows = ops.convert_to_tensor(nrows, value_rowids.dtype, 'nrows')
            const_nrows = tensor_util.constant_value(nrows)
            if const_nrows is not None:
                if const_nrows < 0:
                    raise ValueError('Expected nrows >= 0; got %d' % const_nrows)
                const_rowids = tensor_util.constant_value(value_rowids)
                if const_rowids is not None and const_rowids.size > 0:
                    if not const_nrows >= const_rowids[-1] + 1:
                        raise ValueError('Expected nrows >= value_rowids[-1] + 1; got nrows=%d, value_rowids[-1]=%d' % (const_nrows, const_rowids[-1]))
        value_rowids.shape.assert_has_rank(1)
        nrows.shape.assert_has_rank(0)
        if validate:
            msg = 'Arguments to from_value_rowids do not form a valid RowPartition'
            checks = [check_ops.assert_rank(value_rowids, 1, message=msg), check_ops.assert_rank(nrows, 0, message=msg), check_ops.assert_non_negative(value_rowids[:1], message=msg), _assert_monotonic_increasing(value_rowids, message=msg), check_ops.assert_less(value_rowids[-1:], nrows, message=msg)]
            value_rowids = control_flow_ops.with_dependencies(checks, value_rowids)
        value_rowids_int32 = math_ops.cast(value_rowids, dtypes.int32)
        nrows_int32 = math_ops.cast(nrows, dtypes.int32)
        row_lengths = bincount_ops.bincount(value_rowids_int32, minlength=nrows_int32, maxlength=nrows_int32, dtype=value_rowids.dtype)
        row_splits = array_ops.concat([[0], math_ops.cumsum(row_lengths)], axis=0)
        if const_nrows is not None:
            row_lengths.set_shape([const_nrows])
            row_splits.set_shape([const_nrows + 1])
        return cls(row_splits=row_splits, row_lengths=row_lengths, value_rowids=value_rowids, nrows=nrows, internal=_row_partition_factory_key)