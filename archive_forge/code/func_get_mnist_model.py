import threading
import unittest
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.optimizers.legacy import gradient_descent
from tensorflow.python.distribute.cluster_resolver import (
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.server_lib import (
def get_mnist_model(input_shape):
    """Define a deterministically-initialized CNN model for MNIST testing."""
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer=keras.initializers.TruncatedNormal(seed=99))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Flatten()(x) + keras.layers.Flatten()(x)
    x = keras.layers.Dense(10, activation='softmax', kernel_initializer=keras.initializers.TruncatedNormal(seed=99))(x)
    model = keras.Model(inputs=inputs, outputs=x)
    model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=gradient_descent.SGD(learning_rate=0.001), metrics=['accuracy'])
    return model