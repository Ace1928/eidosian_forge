import warnings
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.losses.loss import Loss
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.saving import serialization_lib
from keras.src.utils.numerical_utils import normalize
@keras_export(['keras.metrics.categorical_focal_crossentropy', 'keras.losses.categorical_focal_crossentropy'])
def categorical_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False, label_smoothing=0.0, axis=-1):
    """Computes the categorical focal crossentropy loss.

    Args:
        y_true: Tensor of one-hot true targets.
        y_pred: Tensor of predicted targets.
        alpha: A weight balancing factor for all classes, default is `0.25` as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using `compute_class_weight` from `sklearn.utils`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple examples in a smooth manner. When `gamma` = 0, there is
            no focal effect on the categorical crossentropy.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability
            distribution.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
            example, if `0.1`, use `0.1 / num_classes` for non-target labels
            and `0.9 + 0.1 / num_classes` for target labels.
        axis: Defaults to `-1`. The dimension along which the entropy is
            computed.

    Returns:
        Categorical focal crossentropy loss value.

    Example:

    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.9, 0.05], [0.1, 0.85, 0.05]]
    >>> loss = keras.losses.categorical_focal_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([2.63401289e-04, 6.75912094e-01], dtype=float32)
    """
    if isinstance(axis, bool):
        raise ValueError(f'`axis` must be of type `int`. Received: axis={axis} of type {type(axis)}')
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)
    if y_pred.shape[-1] == 1:
        warnings.warn(f"In loss categorical_focal_crossentropy, expected y_pred.shape to be (batch_size, num_classes) with num_classes > 1. Received: y_pred.shape={y_pred.shape}. Consider using 'binary_crossentropy' if you only have 2 classes.", SyntaxWarning, stacklevel=2)
    if label_smoothing:
        num_classes = ops.cast(ops.shape(y_true)[-1], y_pred.dtype)
        y_true = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
    if from_logits:
        y_pred = ops.softmax(y_pred, axis=axis)
    output = y_pred / ops.sum(y_pred, axis=axis, keepdims=True)
    output = ops.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
    cce = -y_true * ops.log(output)
    modulating_factor = ops.power(1.0 - output, gamma)
    weighting_factor = ops.multiply(modulating_factor, alpha)
    focal_cce = ops.multiply(weighting_factor, cce)
    focal_cce = ops.sum(focal_cce, axis=axis)
    return focal_cce