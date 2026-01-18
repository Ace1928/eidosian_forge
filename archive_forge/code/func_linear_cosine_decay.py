import functools
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.linear_cosine_decay'])
def linear_cosine_decay(learning_rate, global_step, decay_steps, num_periods=0.5, alpha=0.0, beta=0.001, name=None):
    """Applies linear cosine decay to the learning rate.

  Note that linear cosine decay is more aggressive than cosine decay and
  larger initial learning rates can typically be used.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies a linear cosine decay function
  to a provided initial learning rate.  It requires a `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:
  ```python
  global_step = min(global_step, decay_steps)
  linear_decay = (decay_steps - global_step) / decay_steps)
  cosine_decay = 0.5 * (
      1 + cos(pi * 2 * num_periods * global_step / decay_steps))
  decayed = (alpha + linear_decay) * cosine_decay + beta
  decayed_learning_rate = learning_rate * decayed
  ```

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed = linear_cosine_decay(learning_rate, global_step, decay_steps)
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` Tensor or a Python number.
      The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number. Global
      step to use for the decay computation.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number. Number
      of steps to decay over.
    num_periods: Number of periods in the cosine part of the decay. See
      computation above.
    alpha: See computation above.
    beta: See computation above.
    name: String.  Optional name of the operation.  Defaults to
      'LinearCosineDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  Raises:
    ValueError: if `global_step` is not supplied.

  References:
    Neural Optimizer Search with Reinforcement Learning:
      [Bello et al., 2017](http://proceedings.mlr.press/v70/bello17a.html)
      ([pdf](http://proceedings.mlr.press/v70/bello17a/bello17a.pdf))
    Stochastic Gradient Descent with Warm Restarts:
      [Loshchilov et al., 2017]
      (https://openreview.net/forum?id=Skq89Scxx&noteId=Skq89Scxx)
      ([pdf](https://openreview.net/pdf?id=Skq89Scxx))

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  """
    decayed_lr = learning_rate_schedule.LinearCosineDecay(learning_rate, decay_steps, num_periods=num_periods, alpha=alpha, beta=beta, name=name)
    if not context.executing_eagerly():
        decayed_lr = decayed_lr(global_step)
    else:
        decayed_lr = functools.partial(decayed_lr, global_step)
    return decayed_lr