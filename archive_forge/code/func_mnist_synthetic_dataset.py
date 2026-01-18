import threading
import unittest
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.optimizers.legacy import gradient_descent
from tensorflow.python.distribute.cluster_resolver import (
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.server_lib import (
def mnist_synthetic_dataset(batch_size, steps_per_epoch, target_values='constant'):
    """Generate synthetic MNIST dataset for testing."""
    x_train = tf.ones([batch_size * steps_per_epoch, 28, 28, 1], dtype=tf.float32)
    if target_values == 'constant':
        y_train = tf.ones([batch_size * steps_per_epoch, 1], dtype=tf.int32)
    elif target_values == 'increasing':
        y_train = tf.reshape(tf.range(batch_size * steps_per_epoch, dtype=tf.int32), (-1, 1))
    else:
        raise ValueError('Unknown value for `target_values` "' + str(target_values) + '". Valid options are "constant" and "increasing".')
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    x_test = tf.random.uniform([10000, 28, 28, 1], dtype=tf.float32)
    y_test = tf.random.uniform([10000, 1], minval=0, maxval=9, dtype=tf.int32)
    eval_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    eval_ds = eval_ds.batch(batch_size, drop_remainder=True)
    return (train_ds, eval_ds)