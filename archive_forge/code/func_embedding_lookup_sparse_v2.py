from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('nn.embedding_lookup_sparse', v1=[])
@dispatch.add_dispatch_support
def embedding_lookup_sparse_v2(params, sp_ids, sp_weights, combiner=None, max_norm=None, name=None, allow_fast_lookup=False):
    """Looks up embeddings for the given ids and weights from a list of tensors.

  `params` is a dense tensor or a list of dense tensors, and `sp_ids` is a 2D
  `tf.SparseTensor` or `tf.RaggedTensor` indicating the indices of `params` to
  gather.

  This op is best described with an example. Suppose `params` is an embedding
  table of size `(4, 2)` and `sp_ids` has 3 rows. Since `sp_ids` is sparse or
  ragged, not every row has the same number of elements. The output has shape
  (3, 2). Each row of `sp_ids` is a list of indices, where each index selects a
  row of `params`. For a given row of `sp_ids`, the rows of `params` are
  gathered based on the indices in `sp_ids`, then combined by taking their sum
  or mean.

  >>> params = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
  >>> sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
  ...                          values=[0, 1, 3, 2], dense_shape=(3, 2))
  >>> tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights=None,
  ...                               combiner='sum').numpy()
  array([[4., 6.], [7., 8.], [5., 6.]], dtype=float32)

  In this example, `sp_ids` has 3 rows, so the output has 3 rows. Row 0 of
  `sp_ids` has values 0 and 1, so it selects rows 0 and 1 from `params`, which
  are `[1, 2]` and `[3, 4]`. The rows are summed since `combiner='sum'`,
  resulting in the output row of `[4, 6]`.

  Since row 1 and 2 of `sp_ids` only have one value each, they simply select the
  corresponding row from `params` as the output row. Row 1 has value `3` so
  it selects the `params` elements `[7, 8]` and row 2 has the value 2 so it
  selects the `params` elements `[5, 6]`.

  If `sparse_weights` is specified, it must have the same shape as `sp_ids`.
  `sparse_weights` is used to assign a weight to each slice of `params`. For
  example:

  >>> params = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
  >>> sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
  ...                          values=[0, 1, 3, 2], dense_shape=(3, 2))
  >>> sparse_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
  ...                                  values=[0.1, 1.0, 0.5, 2.0],
  ...                                  dense_shape=(3, 2))
  >>> tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights=sparse_weights,
  ...                               combiner='sum').numpy()
  array([[3.1, 4.2], [3.5, 4.], [10., 12.]], dtype=float32)

  In general, `params` can have shape `(p0, ..., pn)` and `sp_ids` can have `M`
  rows, where each row can have any number of elements. The output has shape
  `(M, p1, ..., pn)`. Each slice of the output `output[i, ...]` is obtained as
  follows: The `combiner` argument is used to combine the values
  `params[sp_ids[i, j], ...] * sparse_weights[i, j]` for each `j` in `range(0,
  len(sp_ids[i]))`, e.g. by taking the sum or mean of the values.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  `sp_ids` and `sp_weights` (if not None) are `SparseTensor`s or `RaggedTensor`s
  with rank of 2. For `SpareTensor`s with left-aligned non-zero entries which
  can be described as `RaggedTensor`s, use of `RaggedTensor`s can yield higher
  performance.

  This op assumes that all id values lie in the range [0, p0), where p0
  is `params.shape[0]`. If you want a version of this op that prunes id values
  less than 0, see `tf.nn.safe_embedding_lookup_sparse`

  If `len(params) > 1`, each element of `sp_ids` is partitioned between the
  elements of `params` according to the "div" partition strategy, which means we
  assign ids to partitions in a contiguous manner. For instance, 13 ids are
  split across 5 partitions as:
  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`.

  If the id space does not evenly divide the number of partitions, each of the
  first `(max_id + 1) % len(params)` partitions will be assigned one more id.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list of tensors all of same shape except for the first dimension,
      representing sharded embedding tensors following "div" partition strategy.
    sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size
      and M is arbitrary or a `RaggedTensor` with rank 2.
    sparse_weights: `SparseTensor` or `RaggedTensor` of same type and shape as
      `sparse_ids`, containing float / double weights corresponding to
      `sparse_ids`, or `None` if all weights are assumed to be 1.0.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
      and "sum" are supported. "sum" computes the weighted sum of the embedding
      results for each row. "mean" is the weighted sum divided by the total
      weight. "sqrtn" is the weighted sum divided by the square root of the sum
      of the squares of the weights. Defaults to `mean`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    name: Optional name for the op.
    allow_fast_lookup: An optional boolean specifying whether to allow
      simplified embedding lookups when `params` is a single tensor and
      `max_norm` is `None`. Setting this flag to `True` during training can
      cause the use of dense gradients with increased memory footprint.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined params) = [p0, p1, ..., pm]`

    and

      `shape(sp_ids) = shape(sp_weights) = [d0, d1]`

    then

      `shape(output) = [d0, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id 0, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```

  Raises:
    TypeError: If `sp_ids` is not a `SparseTensor`, or if `sp_weights` is
      neither `None` nor `SparseTensor`.
    ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
  """
    return embedding_lookup_sparse(params, sp_ids, sp_weights, 'div', name, combiner, max_norm, allow_fast_lookup)