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
@tf_export(v1=['random.stateless_multinomial'])
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions='Use `tf.random.stateless_categorical` instead.')
def stateless_multinomial(logits, num_samples, seed, output_dtype=dtypes.int64, name=None):
    """Draws deterministic pseudorandom samples from a multinomial distribution.

  This is a stateless version of `tf.random.categorical`: if run twice with the
  same seeds and shapes, it will produce the same pseudorandom numbers.  The
  output is consistent across multiple runs on the same hardware (and between
  CPU and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
  hardware.

  Example:

  ```python
  # samples has shape [1, 5], where each value is either 0 or 1 with equal
  # probability.
  samples = tf.random.stateless_categorical(
      tf.math.log([[0.5, 0.5]]), 5, seed=[7, 17])
  ```

  Args:
    logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i,
      :]` represents the unnormalized log-probabilities for all classes.
    num_samples: 0-D.  Number of independent samples to draw for each row slice.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    output_dtype: The integer type of the output: `int32` or `int64`. Defaults
      to `int64`.
    name: Optional name for the operation.

  Returns:
    The drawn samples of shape `[batch_size, num_samples]`.
  """
    with ops.name_scope(name, 'stateless_multinomial', [logits, seed]):
        return stateless_multinomial_categorical_impl(logits, num_samples, output_dtype, seed)