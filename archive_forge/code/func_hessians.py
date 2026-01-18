from tensorflow.python.debug.lib import debug_gradients  # pylint: disable=unused-import
from tensorflow.python.debug.lib import dumping_callback  # pylint: disable=unused-import
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops  # pylint: disable=unused-import
from tensorflow.python.ops import control_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import image_grad  # pylint: disable=unused-import
from tensorflow.python.ops import linalg_grad  # pylint: disable=unused-import
from tensorflow.python.ops import linalg_ops  # pylint: disable=unused-import
from tensorflow.python.ops import logging_ops  # pylint: disable=unused-import
from tensorflow.python.ops import manip_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import optional_grad  # pylint: disable=unused-import
from tensorflow.python.ops import random_grad  # pylint: disable=unused-import
from tensorflow.python.ops import rnn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import sdca_ops  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_grad  # pylint: disable=unused-import
from tensorflow.python.ops.signal import fft_ops  # pylint: disable=unused-import
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.training import checkpoint_ops  # pylint: disable=unused-import
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['hessians'])
def hessians(ys, xs, name='hessians', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None):
    """Constructs the Hessian of sum of `ys` with respect to `x` in `xs`.

  `hessians()` adds ops to the graph to output the Hessian matrix of `ys`
  with respect to `xs`.  It returns a list of `Tensor` of length `len(xs)`
  where each tensor is the Hessian of `sum(ys)`.

  The Hessian is a matrix of second-order partial derivatives of a scalar
  tensor (see https://en.wikipedia.org/wiki/Hessian_matrix for more details).

  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'hessians'.
    colocate_gradients_with_ops: See `gradients()` documentation for details.
    gate_gradients: See `gradients()` documentation for details.
    aggregation_method: See `gradients()` documentation for details.

  Returns:
    A list of Hessian matrices of `sum(ys)` for each `x` in `xs`.

  Raises:
    LookupError: if one of the operations between `xs` and `ys` does not
      have a registered gradient function.
  """
    xs = gradients_util._AsList(xs)
    kwargs = {'colocate_gradients_with_ops': colocate_gradients_with_ops, 'gate_gradients': gate_gradients, 'aggregation_method': aggregation_method}
    hessians = []
    _gradients = gradients(ys, xs, **kwargs)
    for gradient, x in zip(_gradients, xs):
        gradient = array_ops.reshape(gradient, [-1])
        n = array_ops.size(x)
        loop_vars = [array_ops.constant(0, dtypes.int32), tensor_array_ops.TensorArray(x.dtype, n)]
        _, hessian = while_loop.while_loop(lambda j, _: j < n, lambda j, result: (j + 1, result.write(j, gradients(gradient[j], x)[0])), loop_vars)
        _shape = array_ops.shape(x)
        _reshaped_hessian = array_ops.reshape(hessian.stack(), array_ops.concat((_shape, _shape), 0))
        hessians.append(_reshaped_hessian)
    return hessians