from tensorflow.python.ops import gen_collective_ops
def all_reduce_v3(communicator, t, reduction='Add', group_assignment=None, timeout_seconds=None):
    """Reduces tensors mutually.

  Args:
    communicator: the resource `tf.Tensor` returned from
      `initialize_communicator`.
    t: the `tf.Tensor` to be reduced.
    reduction: a string. The name of the operation to reduce the values.
      Accpeted values are `"min"`, `"max"`, `"mul"`, `"add"`.
    group_assignment: Optional int32 `tf.Tensor` with shape [num_groups,
      num_ranks_per_group]. `group_assignment[i]` represents the ranks in the
      `ith` subgroup.
    timeout_seconds: If set to a non zero, set a completion timeout to detect
      staleness. If the timer goes off, a DeadlineExceededError is raised. The
      timeout value in seconds. This feature is experimental.

  Returns:
    The reduced `tf.Tensor`.
  """
    if group_assignment is None:
        group_assignment = []
    return gen_collective_ops.collective_reduce_v3(communicator=communicator, input=t, group_assignment=group_assignment, reduction=reduction, timeout_seconds=timeout_seconds)