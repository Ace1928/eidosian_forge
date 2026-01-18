import functools
import threading
import traceback  # pylint: disable=unused-import
import weakref
import numpy as np
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.lib.core import _pywrap_py_func
from tensorflow.python.ops import autograph_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_script_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def py_func_common(func, inp, Tout, stateful=True, name=None):
    """Wraps a python function and uses it as a TensorFlow op.

  Given a python function `func`, which takes numpy arrays as its
  arguments and returns numpy arrays as its outputs, wrap this function as an
  operation in a TensorFlow graph. The following snippet constructs a simple
  TensorFlow graph that invokes the `np.sinh()` NumPy function as a operation
  in the graph:

  ```python
  def my_func(x):
    # x will be a numpy array with the contents of the placeholder below
    return np.sinh(x)
  input = tf.compat.v1.placeholder(tf.float32)
  y = tf.compat.v1.py_func(my_func, [input], tf.float32)
  ```

  **N.B.** The `tf.compat.v1.py_func()` operation has the following known
  limitations:

  * The body of the function (i.e. `func`) will not be serialized in a
    `GraphDef`. Therefore, you should not use this function if you need to
    serialize your model and restore it in a different environment.

  * The operation must run in the same address space as the Python program
    that calls `tf.compat.v1.py_func()`. If you are using distributed
    TensorFlow, you
    must run a `tf.distribute.Server` in the same process as the program that
    calls
    `tf.compat.v1.py_func()` and you must pin the created operation to a device
    in that
    server (e.g. using `with tf.device():`).

  Note: It produces tensors of unknown shape and rank as shape inference
    does not work on arbitrary Python code.
    If you need the shape, you need to set it based on statically
    available information.

    E.g.
    ```python
    import tensorflow as tf
    import numpy as np

    def make_synthetic_data(i):
        return np.cast[np.uint8](i) * np.ones([20,256,256,3],
                dtype=np.float32) / 10.

    def preprocess_fn(i):
        ones = tf.py_function(make_synthetic_data,[i],tf.float32)
        ones.set_shape(tf.TensorShape([None, None, None, None]))
        ones = tf.image.resize(ones, [224,224])
        return ones

    ds = tf.data.Dataset.range(10)
    ds = ds.map(preprocess_fn)
    ```

  Args:
    func: A Python function, which accepts `ndarray` objects as arguments and
      returns a list of `ndarray` objects (or a single `ndarray`). This function
      must accept as many arguments as there are tensors in `inp`, and these
      argument types will match the corresponding `tf.Tensor` objects in `inp`.
      The returns `ndarray`s must match the number and types defined `Tout`.
      Important Note: Input and output numpy `ndarray`s of `func` are not
        guaranteed to be copies. In some cases their underlying memory will be
        shared with the corresponding TensorFlow tensors. In-place modification
        or storing `func` input or return values in python datastructures
        without explicit (np.)copy can have non-deterministic consequences.
    inp: A list of `Tensor` objects.
    Tout: A list or tuple of tensorflow data types or a single tensorflow data
      type if there is only one, indicating what `func` returns.
    stateful: (Boolean.) If True, the function should be considered stateful. If
      a function is stateless, when given the same input it will return the same
      output and have no observable side effects. Optimizations such as common
      subexpression elimination are only performed on stateless operations.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` or a single `Tensor` which `func` computes.

  @compatibility(TF2)

  This name was deprecated and removed in TF2, but `tf.numpy_function` is a
  near-exact replacement, just drop the `stateful` argument (all
  `tf.numpy_function` calls are considered stateful). It is compatible with
  eager execution and `tf.function`.

  `tf.py_function` is a close but not an exact replacement, passing TensorFlow
  tensors to the wrapped function instead of NumPy arrays, which provides
  gradients and can take advantage of accelerators.

  Before:

  >>> def fn_using_numpy(x):
  ...   x[0] = 0.
  ...   return x
  >>> tf.compat.v1.py_func(fn_using_numpy, inp=[tf.constant([1., 2.])],
  ...     Tout=tf.float32, stateful=False)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 2.], dtype=float32)>

  After:

  >>> tf.numpy_function(fn_using_numpy, inp=[tf.constant([1., 2.])],
  ...     Tout=tf.float32)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 2.], dtype=float32)>

  @end_compatibility

  """
    if context.executing_eagerly():
        result = func(*[np.array(x) for x in inp])
        result = nest.flatten(result)
        result = [x if x is None else ops.convert_to_tensor(x) for x in result]
        if len(result) == 1:
            result, = result
        return result
    if ops.executing_eagerly_outside_functions():
        with ops.device(context.context().host_address_space()):
            return _internal_py_func(func=func, inp=inp, Tout=Tout, stateful=stateful, use_eager_py_func=False, name=name)
    return _internal_py_func(func=func, inp=inp, Tout=Tout, stateful=stateful, use_eager_py_func=False, name=name)