import six
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def sequence_length_from_sparse_tensor(sp_tensor, num_elements=1):
    """Returns a [batch_size] Tensor with per-example sequence length."""
    with ops.name_scope(None, 'sequence_length') as name_scope:
        row_ids = sp_tensor.indices[:, 0]
        column_ids = sp_tensor.indices[:, 1]
        column_ids += array_ops.ones_like(column_ids)
        seq_length = math_ops.segment_max(column_ids, segment_ids=row_ids)
        seq_length = math_ops.cast(math_ops.ceil(seq_length / num_elements), dtypes.int64)
        n_pad = array_ops.shape(sp_tensor)[:1] - array_ops.shape(seq_length)[:1]
        padding = array_ops.zeros(n_pad, dtype=seq_length.dtype)
        return array_ops.concat([seq_length, padding], axis=0, name=name_scope)