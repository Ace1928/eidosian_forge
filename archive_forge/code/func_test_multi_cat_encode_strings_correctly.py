import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from autokeras import keras_layers as layer_module
def test_multi_cat_encode_strings_correctly(tmp_path):
    x_train = np.array([['a', 'ab', 2.1], ['b', 'bc', 1.0], ['a', 'bc', 'nan']])
    layer = layer_module.MultiCategoryEncoding([layer_module.INT, layer_module.INT, layer_module.NONE])
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(32)
    layer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(32))
    for data in dataset:
        result = layer(data)
    assert result[0][0] == result[2][0]
    assert result[0][0] != result[1][0]
    assert result[0][1] != result[1][1]
    assert result[0][1] != result[2][1]
    assert result[2][2] == 0
    assert result.dtype == tf.float32