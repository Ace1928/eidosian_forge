from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
def fully_connected(inp, inp_size, layer_size, name, activation=tf.nn.relu, dtype=tf.dtypes.float32):
    """Helper method to create a fully connected hidden layer."""
    wt = tf.compat.v1.get_variable(name='{}_weight'.format(name), shape=[inp_size, layer_size], dtype=dtype)
    bias = tf.compat.v1.get_variable(name='{}_bias'.format(name), shape=[layer_size], initializer=tf.compat.v1.initializers.zeros())
    output = tf.compat.v1.nn.xw_plus_b(inp, wt, bias)
    if activation is not None:
        assert callable(activation)
        output = activation(output)
    return output