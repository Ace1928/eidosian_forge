import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_random_index_shuffle_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import shape_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('random.stateless_poisson')
@dispatch.add_dispatch_support
def stateless_random_poisson(shape, seed, lam, dtype=dtypes.int32, name=None):
    """Outputs deterministic pseudorandom values from a Poisson distribution.

  The generated values follow a Poisson distribution with specified rate
  parameter.

  This is a stateless version of `tf.random.poisson`: if run twice with the same
  seeds and shapes, it will produce the same pseudorandom numbers. The output is
  consistent across multiple runs on the same hardware, but may change between
  versions of TensorFlow or on non-CPU/GPU hardware.

  A slight difference exists in the interpretation of the `shape` parameter
  between `stateless_poisson` and `poisson`: in `poisson`, the `shape` is always
  prepended to the shape of `lam`; whereas in `stateless_poisson` the shape of
  `lam` must match the trailing dimensions of `shape`.

  Example:

  ```python
  samples = tf.random.stateless_poisson([10, 2], seed=[12, 34], lam=[5, 15])
  # samples has shape [10, 2], where each slice [:, 0] and [:, 1] represents
  # the samples drawn from each distribution

  samples = tf.random.stateless_poisson([7, 5, 2], seed=[12, 34], lam=[5, 15])
  # samples has shape [7, 5, 2], where each slice [:, :, 0] and [:, :, 1]
  # represents the 7x5 samples drawn from each of the two distributions

  rate = tf.constant([[1.], [3.], [5.]])
  samples = tf.random.stateless_poisson([30, 3, 1], seed=[12, 34], lam=rate)
  # samples has shape [30, 3, 1], with 30 samples each of 3x1 distributions.
  ```

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    lam: Tensor. The rate parameter "lambda" of the Poisson distribution. Shape
      must match the rightmost dimensions of `shape`.
    dtype: Dtype of the samples (int or float dtypes are permissible, as samples
      are discrete). Default: int32.
    name: A name for the operation (optional).

  Returns:
    samples: A Tensor of the specified shape filled with random Poisson values.
      For each i, each `samples[..., i]` is an independent draw from the Poisson
      distribution with rate `lam[i]`.
  """
    with ops.name_scope(name, 'stateless_random_poisson', [shape, seed, lam]) as name:
        shape = shape_util.shape_tensor(shape)
        result = gen_stateless_random_ops.stateless_random_poisson(shape, seed=seed, lam=lam, dtype=dtype)
        shape_util.maybe_set_static_shape(result, shape)
        return result