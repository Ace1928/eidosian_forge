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
@tf_export('numpy_function')
@dispatch.add_dispatch_support
def numpy_function(func=None, inp=None, Tout=None, stateful=True, name=None):
    """Wraps a python function and uses it as a TensorFlow op.

  Given a python function `func` wrap this function as an operation in a
  `tf.function`. `func` must take numpy arrays as its arguments and
  return numpy arrays as its outputs.

  There are two ways to use `tf.numpy_function`.

  ### As a decorator

  When using `tf.numpy_function` as a decorator:

  * you must set `Tout`
  * you may set `name`
  * you must not set `func` or `inp`

  >>> @tf.numpy_function(Tout=tf.float32)
  ... def my_numpy_func(x):
  ...   # x will be a numpy array with the contents of the input to the
  ...   # tf.function
  ...   print(f'executing eagerly, {x=}')
  ...   return np.sinh(x)

  The function runs eagerly:

  >>> my_numpy_func(1.0).numpy()
  executing eagerly, x=1.0
  1.17520

  The behavior doesn't change inside a `tf.function`:

  >>> @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
  ... def tf_function(input):
  ...   y = tf.numpy_function(my_numpy_func, [input], tf.float32)
  ...   return y
  >>> tf_function(tf.constant(1.)).numpy()
  executing eagerly, x=array(1.)
  1.17520

  ### Inplace

  This form can be useful if you don't control the function's source,
  but it is harder to read.

  Here is the same function with no decorator:

  >>> def my_func(x):
  ...   # x will be a numpy array with the contents of the input to the
  ...   # tf.function
  ...   print(f'executing eagerly, {x=}')
  ...   return np.sinh(x)

  To run `tf.numpy_function` inplace, pass the function, its inputs, and the
  output type in a single call to `tf.numpy_function`:

  >>> tf.numpy_function(my_func, [tf.constant(1.0)], tf.float32)
  executing eagerly, x=array(1.)
  1.17520

  ### More info

  Comparison to `tf.py_function`:
  `tf.py_function` and `tf.numpy_function` are very similar, except that
  `tf.numpy_function` takes numpy arrays, and not `tf.Tensor`s. If you want the
  function to contain `tf.Tensors`, and have any TensorFlow operations executed
  in the function be differentiable, please use `tf.py_function`.

  Note: We recommend to avoid using `tf.numpy_function` outside of
  prototyping and experimentation due to the following known limitations:

  * Calling `tf.numpy_function` will acquire the Python Global Interpreter Lock
    (GIL) that allows only one thread to run at any point in time. This will
    preclude efficient parallelization and distribution of the execution of the
    program. Therefore, you are discouraged to use `tf.numpy_function` outside
    of prototyping and experimentation.

  * The body of the function (i.e. `func`) will not be serialized in a
    `tf.SavedModel`. Therefore, you should not use this function if you need to
    serialize your model and restore it in a different environment.

  * The operation must run in the same address space as the Python program
    that calls `tf.numpy_function()`. If you are using distributed
    TensorFlow, you must run a `tf.distribute.Server` in the same process as the
    program that calls `tf.numpy_function`  you must pin the created
    operation to a device in that server (e.g. using `with tf.device():`).

  * Currently `tf.numpy_function` is not compatible with XLA. Calling
    `tf.numpy_function` inside `tf.function(jit_compile=True)` will raise an
    error.

  * Since the function takes numpy arrays, you cannot take gradients
    through a numpy_function. If you require something that is differentiable,
    please consider using tf.py_function.

  Args:
    func: A Python function, which accepts `numpy.ndarray` objects as arguments
      and returns a list of `numpy.ndarray` objects (or a single
      `numpy.ndarray`). This function must accept as many arguments as there are
      tensors in `inp`, and these argument types will match the corresponding
      `tf.Tensor` objects in `inp`. The returns `numpy.ndarray`s must match the
      number and types defined `Tout`. Important Note: Input and output
      `numpy.ndarray`s of `func` are not guaranteed to be copies. In some cases
      their underlying memory will be shared with the corresponding TensorFlow
      tensors. In-place modification or storing `func` input or return values in
      python datastructures without explicit (np.)copy can have
      non-deterministic consequences.
    inp: A list of `tf.Tensor` objects.
    Tout: A list or tuple of tensorflow data types or a single tensorflow data
      type if there is only one, indicating what `func` returns.
    stateful: (Boolean.) Setting this argument to False tells the runtime to
      treat the function as stateless, which enables certain optimizations. A
      function is stateless when given the same input it will return the same
      output and have no side effects; its only purpose is to have a return
      value. The behavior for a stateful function with the `stateful` argument
      False is undefined. In particular, caution should be taken when mutating
      the input arguments as this is a stateful operation.
    name: (Optional) A name for the operation.

  Returns:
    * If `func` is `None` this returns a decorator that will ensure the
      decorated function will always run with eager execution even if called
      from a `tf.function`/`tf.Graph`.
    * If used `func` is not `None` this executes `func` with eager execution
      and returns the result: A single or list of `tf.Tensor` which `func`
      computes.
  """
    decorator = _check_args_and_maybe_make_decorator(numpy_function, 'tf.numpy_function', func=func, inp=inp, Tout=Tout, stateful=stateful, name=name)
    if decorator is not None:
        return decorator
    return py_func_common(func, inp, Tout, stateful=stateful, name=name)