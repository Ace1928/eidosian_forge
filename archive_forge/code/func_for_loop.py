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
def for_loop(loop_fn, loop_fn_dtypes, iters, parallel_iterations=None):
    """Runs `loop_fn` `iters` times and stacks the outputs.


  Runs `loop_fn` `iters` times, with input values from 0 to `iters - 1`, and
  stacks corresponding outputs of the different runs.

  Args:
    loop_fn: A function that takes an int32 scalar tf.Tensor object representing
      the iteration number, and returns a possibly nested structure of tensor
      objects. The shape of these outputs should not depend on the input.
    loop_fn_dtypes: dtypes for the outputs of `loop_fn`.
    iters: Number of iterations for which to run `loop_fn`.
    parallel_iterations: The number of iterations that can be dispatched in
      parallel. This knob can be used to control the total memory usage.

  Returns:
    Returns a nested structure of stacked output tensor objects with the same
    nested structure as the output of `loop_fn`.
  """
    flat_loop_fn_dtypes = nest.flatten(loop_fn_dtypes)
    is_none_list = []

    def while_body(i, *ta_list):
        """Body of while loop."""
        fn_conv = autograph.tf_convert(loop_fn, autograph_ctx.control_status_ctx())
        fn_output = nest.flatten(fn_conv(i))
        if len(fn_output) != len(flat_loop_fn_dtypes):
            raise ValueError(f'Number of expected outputs {len(flat_loop_fn_dtypes)}, does not match the number of actual outputs {len(fn_output)} from loop_fn: {loop_fn} with output {fn_output}.')
        outputs = []
        del is_none_list[:]
        is_none_list.extend((x is None for x in fn_output))
        for out, ta in zip(fn_output, ta_list):
            if out is not None:
                ta = ta.write(i, out)
            outputs.append(ta)
        return tuple([i + 1] + outputs)
    if parallel_iterations is not None:
        extra_args = {'parallel_iterations': parallel_iterations}
    else:
        extra_args = {}
    ta_list = while_loop.while_loop(lambda i, *ta: i < iters, while_body, [0] + [tensor_array_ops.TensorArray(dtype.base_dtype, iters) for dtype in flat_loop_fn_dtypes], **extra_args)[1:]
    output = [None if is_none else ta.stack() for ta, is_none in zip(ta_list, is_none_list)]
    assert len(output) in (0, len(flat_loop_fn_dtypes))
    if not output:
        loop_var = array_ops.placeholder_with_default(0, shape=[])
        try:
            loop_fn_out = loop_fn(loop_var)
            out_shapes = [[0] + ops.convert_to_tensor(x).shape for x in nest.flatten(loop_fn_out)]
            output = [array_ops.zeros(out_shapes[i], dt) for i, dt in enumerate(flat_loop_fn_dtypes)]
        except Exception:
            output = [array_ops.zeros([0])]
    return nest.pack_sequence_as(loop_fn_dtypes, output)