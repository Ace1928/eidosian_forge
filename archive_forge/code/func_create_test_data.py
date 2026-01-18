import os
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
import keras.src as keras
from keras.src import callbacks as callbacks_lib
from keras.src.engine import sequential
from keras.src.layers import core as core_layers
from keras.src.layers.preprocessing import string_lookup
from keras.src.optimizers.legacy import gradient_descent
from keras.src.utils import dataset_creator
from tensorflow.python.platform import tf_logging as logging
def create_test_data():
    x = tf.constant([[1.0], [2.0], [3.0], [1.0], [5.0], [1.0]])
    return tf.data.Dataset.from_tensor_slices(x).repeat().batch(2)