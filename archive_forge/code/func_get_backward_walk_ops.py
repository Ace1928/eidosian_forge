from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import object_identity
def get_backward_walk_ops(seed_ops, inclusive=True, within_ops=None, within_ops_fn=None, stop_at_ts=(), control_inputs=False, only_differentiable=False):
    """Do a backward graph walk and return all the visited ops.

  Args:
    seed_ops: an iterable of operations from which the backward graph
      walk starts. If a list of tensors is given instead, the seed_ops are set
      to be the generators of those tensors.
    inclusive: if True the given seed_ops are also part of the resulting set.
    within_ops: an iterable of `tf.Operation` within which the search is
      restricted. If `within_ops` is `None`, the search is performed within
      the whole graph.
    within_ops_fn: if provided, a function on ops that should return True iff
      the op is within the graph traversal. This can be used along within_ops,
      in which case an op is within if it is also in within_ops.
    stop_at_ts: an iterable of tensors at which the graph walk stops.
    control_inputs: if True, control inputs will be used while moving backward.
    only_differentiable: if True, only traverse ops which are differentiable.
      This includes natively differentiable ops, or ops with custom gradients.
  Returns:
    A Python set of all the `tf.Operation` behind `seed_ops`.
  Raises:
    TypeError: if `seed_ops` or `within_ops` cannot be converted to a list of
      `tf.Operation`.
  """
    control_inputs = control_inputs and (not only_differentiable)
    if not is_iterable(seed_ops):
        seed_ops = [seed_ops]
    try:
        first_seed_op = next(iter(seed_ops))
    except StopIteration:
        return []
    if isinstance(first_seed_op, tensor_lib.Tensor):
        ts = make_list_of_t(seed_ops, allow_graph=False)
        seed_ops = get_generating_ops(ts)
    else:
        seed_ops = make_list_of_op(seed_ops, allow_graph=False)
    stop_at_ts = object_identity.ObjectIdentitySet(make_list_of_t(stop_at_ts))
    seed_ops = object_identity.ObjectIdentitySet(make_list_of_op(seed_ops))
    if within_ops:
        within_ops = make_list_of_op(within_ops, allow_graph=False)
        within_ops = object_identity.ObjectIdentitySet(within_ops)
        seed_ops &= within_ops

    def is_within(op):
        return (within_ops is None or op in within_ops) and (within_ops_fn is None or within_ops_fn(op))
    result = list(seed_ops)
    wave = set(seed_ops)
    while wave:
        new_wave = set()
        for op in wave:
            for new_t in _get_inputs(op, only_differentiable=only_differentiable):
                if new_t in stop_at_ts:
                    continue
                if new_t.op not in result and is_within(new_t.op):
                    new_wave.add(new_t.op)
            if control_inputs:
                for new_op in op.control_inputs:
                    if new_op not in result and is_within(new_op):
                        new_wave.add(new_op)
        concatenate_unique(result, new_wave)
        wave = new_wave
    if not inclusive:
        result = [op for op in result if op not in seed_ops]
    return result