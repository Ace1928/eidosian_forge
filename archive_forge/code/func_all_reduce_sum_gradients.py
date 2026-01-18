from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.ops import clip_ops
from tensorflow.python.platform import tf_logging as logging
def all_reduce_sum_gradients(grads_and_vars):
    """Returns all-reduced gradients aggregated via summation.

  Args:
    grads_and_vars: List of (gradient, variable) pairs.

  Returns:
    List of (gradient, variable) pairs where gradients have been all-reduced.
  """
    grads_and_vars = list(grads_and_vars)
    filtered_grads_and_vars = filter_empty_gradients(grads_and_vars)
    if filtered_grads_and_vars:
        if strategy_supports_no_merge_call():
            grads = [pair[0] for pair in filtered_grads_and_vars]
            reduced = distribute_lib.get_strategy().extended._replica_ctx_all_reduce(ds_reduce_util.ReduceOp.SUM, grads)
        else:
            reduced = distribute_lib.get_replica_context().merge_call(_all_reduce_sum_fn, args=(filtered_grads_and_vars,))
    else:
        reduced = []
    reduced_with_nones = []
    reduced_pos = 0
    for g, v in grads_and_vars:
        if g is None:
            reduced_with_nones.append((None, v))
        else:
            reduced_with_nones.append((reduced[reduced_pos], v))
            reduced_pos += 1
    assert reduced_pos == len(reduced), 'Failed to add all gradients'
    return reduced_with_nones