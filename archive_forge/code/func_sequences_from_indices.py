import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
def sequences_from_indices(array, indices_ds, start_index, end_index):
    dataset = tf.data.Dataset.from_tensors(array[start_index:end_index])
    dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(lambda steps, inds: tf.gather(steps, inds), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset