import numpy as np
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import layers
from keras.src import losses
from keras.src import models
from keras.src.datasets import mnist
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.dtensor import layout_map as layout_map_lib
from keras.src.utils import np_utils
def get_mnist_datasets(num_class, batch_size):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32')
    x_test = np.expand_dims(x_test, axis=-1).astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train, num_class)
    y_test = np_utils.to_categorical(y_test, num_class)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().batch(batch_size, drop_remainder=True)
    eval_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).repeat().batch(batch_size, drop_remainder=True)
    return (train_ds, eval_ds)