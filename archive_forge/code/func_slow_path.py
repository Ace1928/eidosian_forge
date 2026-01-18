import functools
import itertools
import operator
import tensorflow as tf
from keras.src.backend.tensorflow.core import convert_to_tensor
def slow_path():
    all_valid = functools.reduce(operator.and_, validities)
    return tf.where(all_valid, tf.transpose(tf.gather_nd(input_arr, indices)), fill_value)