import itertools
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect
def cholesky(lin_op_a, name=None):
    """Get the Cholesky factor associated to lin_op_a.

  Args:
    lin_op_a: The LinearOperator to decompose.
    name: Name to use for this operation.

  Returns:
    A LinearOperator that represents the lower Cholesky factor of `lin_op_a`.

  Raises:
    NotImplementedError: If no Cholesky method is defined for the LinearOperator
      type of `lin_op_a`.
  """
    cholesky_fn = _registered_cholesky(type(lin_op_a))
    if cholesky_fn is None:
        raise ValueError('No cholesky decomposition registered for {}'.format(type(lin_op_a)))
    with ops.name_scope(name, 'Cholesky'):
        return cholesky_fn(lin_op_a)