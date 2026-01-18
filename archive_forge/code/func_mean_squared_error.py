from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['losses.mean_squared_error'])
@dispatch.add_dispatch_support
def mean_squared_error(labels, predictions, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Adds a Sum-of-Squares loss to the training procedure.

  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  `[batch_size]`, then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.

  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.

  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.

  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or `predictions`
      is None.

  @compatibility(TF2)

  `tf.compat.v1.losses.mean_squared_error` is mostly compatible with eager
  execution and `tf.function`. But, the `loss_collection` argument is
  ignored when executing eagerly and no loss will be written to the loss
  collections. You will need to either hold on to the return value manually
  or rely on `tf.keras.Model` loss tracking.


  To switch to native TF2 style, instantiate the
   `tf.keras.losses.MeanSquaredError` class and call the object instead.


  #### Structural Mapping to Native TF2

  Before:

  ```python
  loss = tf.compat.v1.losses.mean_squared_error(
    labels=labels,
    predictions=predictions,
    weights=weights,
    reduction=reduction)
  ```

  After:

  ```python
  loss_fn = tf.keras.losses.MeanSquaredError(
    reduction=reduction)
  loss = loss_fn(
    y_true=labels,
    y_pred=predictions,
    sample_weight=weights)
  ```

  #### How to Map Arguments

  | TF1 Arg Name          | TF2 Arg Name     | Note                       |
  | :-------------------- | :--------------- | :------------------------- |
  | `labels`              | `y_true`         | In `__call__()` method     |
  | `predictions`         | `y_pred`         | In `__call__()` method     |
  | `weights`             | `sample_weight`  | In `__call__()` method.    |
  : : : The shape requirements for `sample_weight` is different from      :
  : : : `weights`. Please check the [argument definition][api_docs] for   :
  : : : details.                                                          :
  | `scope`               | Not supported    | -                          |
  | `loss_collection`     | Not supported    | Losses should be tracked   |
  : : : explicitly or with Keras APIs, for example, [add_loss][add_loss], :
  : : : instead of via collections                                        :
  | `reduction`           | `reduction`      | In constructor. Value of   |
  : : : `tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE`,              :
  : : : `tf.compat.v1.losses.Reduction.SUM`,                              :
  : : : `tf.compat.v1.losses.Reduction.NONE` in                           :
  : : : `tf.compat.v1.losses.softmax_cross_entropy` correspond to         :
  : : : `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE`,                  :
  : : : `tf.keras.losses.Reduction.SUM`,                                  :
  : : : `tf.keras.losses.Reduction.NONE`, respectively. If you            :
  : : : used other value for `reduction`, including the default value     :
  : : :  `tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS`, there is :
  : : : no directly corresponding value. Please modify the loss           :
  : : : implementation manually.                                          :

  [add_loss]:https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_loss
  [api_docs]:https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError#__call__


  #### Before & After Usage Example

  Before:

  >>> y_true = [1, 2, 3]
  >>> y_pred = [1, 3, 5]
  >>> weights = [0, 1, 0.25]
  >>> # samples with zero-weight are excluded from calculation when `reduction`
  >>> # argument is set to default value `Reduction.SUM_BY_NONZERO_WEIGHTS`
  >>> tf.compat.v1.losses.mean_squared_error(
  ...    labels=y_true,
  ...    predictions=y_pred,
  ...    weights=weights).numpy()
  1.0

  >>> tf.compat.v1.losses.mean_squared_error(
  ...    labels=y_true,
  ...    predictions=y_pred,
  ...    weights=weights,
  ...    reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE).numpy()
  0.66667

  After:

  >>> y_true = [[1.0], [2.0], [3.0]]
  >>> y_pred = [[1.0], [3.0], [5.0]]
  >>> weights = [1, 1, 0.25]
  >>> mse = tf.keras.losses.MeanSquaredError(
  ...    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
  >>> mse(y_true=y_true, y_pred=y_pred, sample_weight=weights).numpy()
  0.66667

  @end_compatibility
  """
    if labels is None:
        raise ValueError('Argument `labels` must not be None.')
    if predictions is None:
        raise ValueError('Argument `predictions` must not be None.')
    with ops.name_scope(scope, 'mean_squared_error', (predictions, labels, weights)) as scope:
        predictions = math_ops.cast(predictions, dtype=dtypes.float32)
        labels = math_ops.cast(labels, dtype=dtypes.float32)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = math_ops.squared_difference(predictions, labels)
        return compute_weighted_loss(losses, weights, scope, loss_collection, reduction=reduction)