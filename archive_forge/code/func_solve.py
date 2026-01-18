import itertools
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect
def solve(lin_op_a, lin_op_b, name=None):
    """Compute lin_op_a.solve(lin_op_b).

  Args:
    lin_op_a: The LinearOperator on the left.
    lin_op_b: The LinearOperator on the right.
    name: Name to use for this operation.

  Returns:
    A LinearOperator that represents the solve between `lin_op_a` and
      `lin_op_b`.

  Raises:
    NotImplementedError: If no solve method is defined between types of
      `lin_op_a` and `lin_op_b`.
  """
    solve_fn = _registered_solve(type(lin_op_a), type(lin_op_b))
    if solve_fn is None:
        raise ValueError('No solve registered for {}.solve({})'.format(type(lin_op_a), type(lin_op_b)))
    with ops.name_scope(name, 'Solve'):
        return solve_fn(lin_op_a, lin_op_b)