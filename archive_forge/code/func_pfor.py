import functools
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.parallel_for.pfor import PFor
from tensorflow.python.ops.parallel_for.pfor import PForConfig
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def pfor(loop_fn, iters, fallback_to_while_loop=True, parallel_iterations=None, warn=False):
    """Equivalent to running `loop_fn` `iters` times and stacking the outputs.

  `pfor` has functionality similar to `for_loop`, i.e. running `loop_fn` `iters`
  times, with input from 0 to `iters - 1`, and stacking corresponding output of
  each iteration. However the implementation does not use a `tf.while_loop`.
  Instead it adds new operations to the graph that collectively compute the same
  value as what running `loop_fn` in a loop would compute.


  This is an experimental feature and currently has a lot of limitations:
    - There should be no data dependency between the different iterations. For
      example, a future iteration should not depend on a value or side-effect of
      a previous iteration.
    - Stateful kernels may mostly not be supported since these often imply a
      data dependency or ordering of the iterations. We do support a limited set
      of such stateful kernels though (like RandomFoo, Variable operations like
      reads, etc).
    - Conversion works only on a limited set of kernels for which a converter
      has been registered.
    - `loop_fn` has limited support for control flow operations. `tf.cond` in
      particular is not supported.
    - `loop_fn` should return nested structure of Tensors or Operations. However
      if an Operation is returned, it should have zero outputs.
    - The shape and dtype of `loop_fn` outputs should not depend on the input
      to loop_fn.

  Args:
    loop_fn: A function that takes an int32 scalar tf.Tensor object representing
      the iteration number, and optionally a keyword argument `pfor_config` set
      to a PForConfig object. It returns a possibly nested structure of Tensor
      or Operation objects. Note that if setting `parallel_iterations` argument
      to something other than None, `loop_fn` may be called more than once
      during graph construction. So it may need to avoid mutating global state.
    iters: Number of iterations for which to run `loop_fn`.
    fallback_to_while_loop: If true, on failing to vectorize an operation, pfor
      fallbacks to using a `tf.while_loop` to dispatch the iterations.
    parallel_iterations: A knob to control how many iterations are vectorized
      and dispatched in parallel. The default value of None corresponds to
      vectorizing all the iterations.  If `parallel_iterations` is smaller than
      `iters`, then chunks of at most that many iterations are dispatched in
      sequence. This knob can be used to control the total memory usage.
    warn: Whether or not to warn when falling back to while loops.

  Returns:
    Returns a nested structure of stacked tensor objects with the same nested
    structure as the output of `loop_fn`.
  Raises:
    ValueError: If parallel_iterations is not None and not an integer > 1.
  """

    def f():
        return _pfor_impl(loop_fn, iters, fallback_to_while_loop=fallback_to_while_loop, parallel_iterations=parallel_iterations, warn=warn)
    functions_run_eagerly = None
    if context.executing_eagerly() or _is_under_xla_context():
        functions_run_eagerly = def_function.functions_run_eagerly()
        if functions_run_eagerly:
            logging.warning('It looks like tf.function behavior was disabled, perhaps using tf.config.run_functions_eagerly. Vectorization primitives (e.g. tf.vectorized_map) require tf.function to work. These primitives will override the disable.')
            def_function.run_functions_eagerly(False)
        f = def_function.function(f)
    outputs = f()
    if functions_run_eagerly is not None:
        def_function.run_functions_eagerly(functions_run_eagerly)
    return outputs