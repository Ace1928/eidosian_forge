import uuid
import tensorflow.compat.v2 as tf
from tensorflow.python.eager.context import get_device_name
def has_fully_masked_sequence(mask):
    return tf.reduce_any(tf.reduce_all(tf.logical_not(mask), axis=1))