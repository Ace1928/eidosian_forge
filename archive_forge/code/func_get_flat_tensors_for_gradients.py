import abc
import sys
from tensorflow.python.framework import composite_tensor
from tensorflow.python.util import nest
def get_flat_tensors_for_gradients(xs):
    """Returns a flat list of Tensors that should be differentiated for `xs`.

  Args:
    xs: A list of `Tensor`s or `CompositeTensor`s.

  Returns:
    A flat list of `Tensor`s constructed from `xs`, where `Tensor` values are
    left as-is, and `CompositeTensor`s are replaced with
    `_get_tensors_for_gradient(x)`.
  """
    return nest.flatten([_get_tensors_for_gradient(x) for x in xs])