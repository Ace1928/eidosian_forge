from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import object_identity
def make_list_of_t(ts, check_graph=True, allow_graph=True, ignore_ops=False):
    """Convert ts to a list of `tf.Tensor`.

  Args:
    ts: can be an iterable of `tf.Tensor`, a `tf.Graph` or a single tensor.
    check_graph: if `True` check if all the tensors belong to the same graph.
    allow_graph: if `False` a `tf.Graph` cannot be converted.
    ignore_ops: if `True`, silently ignore `tf.Operation`.
  Returns:
    A newly created list of `tf.Tensor`.
  Raises:
    TypeError: if `ts` cannot be converted to a list of `tf.Tensor` or,
     if `check_graph` is `True`, if all the ops do not belong to the same graph.
  """
    if isinstance(ts, ops.Graph):
        if allow_graph:
            return get_tensors(ts)
        else:
            raise TypeError('allow_graph is False: cannot convert a tf.Graph.')
    else:
        if not is_iterable(ts):
            ts = [ts]
        if not ts:
            return []
        if check_graph:
            check_types = None if ignore_ops else tensor_lib.Tensor
            get_unique_graph(ts, check_types=check_types)
        return [t for t in ts if isinstance(t, tensor_lib.Tensor)]