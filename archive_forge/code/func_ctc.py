import warnings
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.losses.loss import Loss
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.saving import serialization_lib
from keras.src.utils.numerical_utils import normalize
@keras_export('keras.losses.ctc')
def ctc(y_true, y_pred):
    """CTC (Connectionist Temporal Classification) loss.

    Args:
        y_true: A tensor of shape `(batch_size, max_length)` containing
            the true labels in integer format. `0` always represents
            the blank/mask index and should not be used for classes.
        y_pred: A tensor of shape `(batch_size, max_length, num_classes)`
            containing logits (the output of your model).
            They should *not* be normalized via softmax.
    """
    if len(ops.shape(y_true)) != 2:
        raise ValueError(f'Targets `y_true` are expected to be a tensor of shape `(batch_size, max_length)` in integer format. Received: y_true.shape={ops.shape(y_true)}')
    if len(ops.shape(y_pred)) != 3:
        raise ValueError(f'Logits `y_pred` are expected to be a tensor of shape `(batch_size, max_length, num_classes)`. Received: y_pred.shape={ops.shape(y_pred)}')
    batch_length = ops.cast(ops.shape(y_true)[0], dtype='int32')
    input_length = ops.cast(ops.shape(y_pred)[1], dtype='int32')
    label_length = ops.cast(ops.shape(y_true)[1], dtype='int32')
    input_length = input_length * ops.ones((batch_length,), dtype='int32')
    label_length = label_length * ops.ones((batch_length,), dtype='int32')
    return ops.ctc_loss(y_true, y_pred, label_length, input_length, mask_index=0)