from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator
def ragged_abs(self, name=None):
    """Computes the absolute value of a ragged tensor.

  Given a ragged tensor of integer or floating-point values, this operation
  returns a ragged tensor of the same type, where each element contains the
  absolute value of the corresponding element in the input.

  Given a ragged tensor `x` of complex numbers, this operation returns a tensor
  of type `float32` or `float64` that is the absolute value of each element in
  `x`. For a complex number \\\\(a + bj\\\\), its absolute value is computed as
  \\\\(\\sqrt{a^2 + b^2}\\\\).

  For example:

  >>> # real number
  >>> x = tf.ragged.constant([[-2.2, 3.2], [-4.2]])
  >>> tf.abs(x)
  <tf.RaggedTensor [[2.2, 3.2], [4.2]]>

  >>> # complex number
  >>> x = tf.ragged.constant([[-2.2 + 4.7j], [-3.2 + 5.7j], [-4.2 + 6.7j]])
  >>> tf.abs(x)
  <tf.RaggedTensor [[5.189412298131649],
   [6.536818798161687],
   [7.907591289387685]]>

  Args:
    name: A name for the operation (optional).

  Returns:
    A `RaggedTensor` of the same size and type as `x`, with absolute values.
    Note, for `complex64` or `complex128` input, the returned `RaggedTensor`
    will be of type `float32` or `float64`, respectively.
  """
    return math_ops.abs(self, name=name)