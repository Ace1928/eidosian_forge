import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['sparse.placeholder', 'sparse_placeholder'])
@deprecation.deprecated_endpoints('sparse_placeholder')
def sparse_placeholder(dtype, shape=None, name=None):
    """Inserts a placeholder for a sparse tensor that will be always fed.

  **Important**: This sparse tensor will produce an error if evaluated.
  Its value must be fed using the `feed_dict` optional argument to
  `Session.run()`, `Tensor.eval()`, or `Operation.run()`.

  For example:

  ```python
  x = tf.compat.v1.sparse.placeholder(tf.float32)
  y = tf.sparse.reduce_sum(x)

  with tf.compat.v1.Session() as sess:
    print(sess.run(y))  # ERROR: will fail because x was not fed.

    indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float32)
    shape = np.array([7, 9, 2], dtype=np.int64)
    print(sess.run(y, feed_dict={
      x: tf.compat.v1.SparseTensorValue(indices, values, shape)}))  # Will
      succeed.
    print(sess.run(y, feed_dict={
      x: (indices, values, shape)}))  # Will succeed.

    sp = tf.sparse.SparseTensor(indices=indices, values=values,
                                dense_shape=shape)
    sp_value = sp.eval(session=sess)
    print(sess.run(y, feed_dict={x: sp_value}))  # Will succeed.
  ```


  Args:
    dtype: The type of `values` elements in the tensor to be fed.
    shape: The shape of the tensor to be fed (optional). If the shape is not
      specified, you can feed a sparse tensor of any shape.
    name: A name for prefixing the operations (optional).

  Returns:
    A `SparseTensor` that may be used as a handle for feeding a value, but not
    evaluated directly.

  Raises:
    RuntimeError: if eager execution is enabled

  @compatibility(TF2)
  This API is not compatible with eager execution and `tf.function`. To migrate
  to TF2, rewrite the code to be compatible with eager execution. Check the
  [migration
  guide](https://www.tensorflow.org/guide/migrate#1_replace_v1sessionrun_calls)
  on replacing `Session.run` calls. In TF2, you can just pass tensors directly
  into ops and layers. If you want to explicitly set up your inputs, also see
  [Keras functional API](https://www.tensorflow.org/guide/keras/functional) on
  how to use `tf.keras.Input` to replace `tf.compat.v1.sparse_placeholder`.
  `tf.function` arguments also do the job of `tf.compat.v1.sparse_placeholder`.
  For more details please read [Better
  performance with tf.function](https://www.tensorflow.org/guide/function).
  @end_compatibility
  """
    if context.executing_eagerly():
        raise RuntimeError('`sparse_placeholder` is not compatible with eager execution.')
    shape_name = name + '/shape' if name is not None else None
    default_shape_name = name + '/shape_default' if name is not None else None
    if shape is None:
        rank = None
        dense_shape = placeholder(dtypes.int64, shape=[rank], name=shape_name)
        dense_shape_default = tensor_util.constant_value_as_shape(dense_shape)
    else:
        if isinstance(shape, tensor_lib.Tensor):
            rank = shape.get_shape()[0]
            dense_shape_default = tensor_util.constant_value_as_shape(shape)
        else:
            rank = len(shape)
            dense_shape_default = tensor_shape.TensorShape(tuple((None if dim == -1 else dim for dim in shape)))
            shape = tuple((tensor_shape.dimension_value(dim) for dim in shape))
            shape = tuple((-1 if dim is None else dim for dim in shape))
            shape = ops.convert_to_tensor(shape, dtype=dtypes.int64, name=default_shape_name)
        dense_shape = placeholder_with_default(shape, shape=shape.shape, name=shape_name)
    result = sparse_tensor.SparseTensor(values=placeholder(dtype, shape=[None], name=name + '/values' if name is not None else None), indices=placeholder(dtypes.int64, shape=[None, rank], name=name + '/indices' if name is not None else None), dense_shape=dense_shape)
    result.set_shape(dense_shape_default)
    return result