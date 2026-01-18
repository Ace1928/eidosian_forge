import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
def noised():
    return inputs + self._random_generator.random_normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev, dtype=inputs.dtype)