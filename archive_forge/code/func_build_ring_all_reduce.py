import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def build_ring_all_reduce(input_tensors, num_workers, num_subchunks, gpu_perm, red_op, un_op=None):
    """Construct a subgraph performing a ring-style all-reduce of input_tensors.

  Args:
    input_tensors: a list of `tf.Tensor` objects, which must all
      have the same shape and type.
    num_workers: number of worker tasks spanned by input_tensors.
    num_subchunks: number of subchunks each device should process in one tick.
    gpu_perm: a list of ints giving a ring-wise rank ordering of GPUs at
      each worker.  All workers must have the same number of
      GPUs with the same rank ordering.  If NVLINK is available, this should
      be a ring order supported by NVLINK edges.
    red_op: a binary operator for elementwise reduction.
    un_op: an optional unary operator to apply to fully reduced values.

  Raises:
    ValueError: empty input_tensors or they don't all have same
    size.

  Returns:
    a list of `tf.Tensor` identical sum-reductions of input_tensors.
  """
    if len(input_tensors) < 2:
        raise ValueError('input_tensors must be length 2 or longer')
    input_tensors, shape = _flatten_tensors(input_tensors)
    devices = [t.device for t in input_tensors]
    pred_by_s_d, rank_by_s_d = _ring_permutations(num_workers, num_subchunks, gpu_perm)
    chunks_by_dev, pad_len = _build_ring_gather(input_tensors, devices, num_subchunks, pred_by_s_d, rank_by_s_d, red_op)
    if un_op:
        chunks_by_dev = _apply_unary_to_chunks(un_op, chunks_by_dev)
    output_tensors = _build_ring_scatter(pred_by_s_d, rank_by_s_d, chunks_by_dev)
    if pad_len > 0:
        output_tensors = _strip_padding(output_tensors, pad_len)
    if len(shape) != 1:
        output_tensors = _reshape_tensors(output_tensors, shape)
    return output_tensors