import tree
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
def class_weights_map_fn(*data):
    """Convert `class_weight` to `sample_weight`."""
    x, y, sw = data_adapter_utils.unpack_x_y_sample_weight(data)
    if sw is not None:
        raise ValueError('You cannot `class_weight` and `sample_weight` at the same time.')
    if tree.is_nested(y):
        raise ValueError('`class_weight` is only supported for Models with a single output.')
    if y.shape.rank >= 2:
        y_classes = tf.__internal__.smart_cond.smart_cond(tf.shape(y)[-1] > 1, lambda: tf.argmax(y, axis=-1), lambda: tf.cast(tf.round(tf.squeeze(y, axis=-1)), tf.int32))
    else:
        y_classes = tf.cast(tf.round(y), tf.int32)
    cw = tf.gather(class_weight_tensor, y_classes)
    return (x, y, cw)