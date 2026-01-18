import uuid
import tensorflow.compat.v2 as tf
from tensorflow.python.eager.context import get_device_name
def runtime(runtime_name):
    with tf.device('/cpu:0'):
        return tf.constant(runtime_name, dtype=tf.float32, name='runtime')