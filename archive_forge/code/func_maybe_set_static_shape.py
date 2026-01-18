from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
def maybe_set_static_shape(tensor, shape):
    """Sets the shape of `tensor` to the `shape`'s constant value, if inferrable.

  This is a temporary workaround to fix shape inference across functional op
  boundaries. E.g.

  ```python
  shape = tf.constant([3])
  @tf.function
  def f():
    u = tf.random_uniform(shape)
    return u
  ```

  If we were to rely solely on C++ shape inference, the shape of `u` inside
  `f` would be unknown because C++ shape inference is not aware of the outer
  graph and all it sees is a Placeholder node when backtracing the captured
  tensor for `shape`. `maybe_set_static_shape` computes the static shape value
  of `shape` by traversing the `FuncGraph` boundaries and sets the correct
  shape.

  A longer term solution would be to fix C++ shape inference.

  Args:
    tensor: A tensor.
    shape: A shape tensor.
  """
    if _ENABLE_MAYBE_SET_STATIC_SHAPE and (not context.executing_eagerly()) and ops.get_default_graph().building_function and (not tensor.shape.is_fully_defined()) and tensor_util.is_tensor(shape):
        shape = shape_tensor(shape)
        const_shape = tensor_util.constant_value_as_shape(shape)
        tensor.set_shape(const_shape)