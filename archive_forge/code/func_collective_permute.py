from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_tpu_ops
from tensorflow.python.ops.gen_tpu_ops import *
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.util.tf_export import tf_export
def collective_permute(x, source_target_pairs, name=None):
    """Permute the input tensor across replicas given source_target_pairs.

  For each source_target_pair <a, b>, we send replica a's input to replica b.
  Each replica id must only appear once in the source column. Also it must
  only appear once in the target column.
  For the replica id not in the target column, this op returns a zero tensor
  with the same shape and dtype of the input x.

  For example, suppose there are 4 TPU instances: `[A, B, C, D]`. Passing
  source_target_pairs=`[[0,1],[1,2],[2,3]]` gets the outputs:
  `[0, A, B, C]`.

  Args:
    x: The local tensor to be permuted.
    source_target_pairs: 2d int lists with shape [num_pairs, 2].
      source_target_pairs[i][0] represents the source replica id and
      source_target_pairs[i][1] represents the target replica id.
    name: Optional op name.

  Returns:
    A `Tensor` which is permuted.
  """
    return gen_tpu_ops.collective_permute(x, source_target_pairs, name=name)