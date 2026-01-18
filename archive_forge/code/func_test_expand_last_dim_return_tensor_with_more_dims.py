import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from autokeras import keras_layers as layer_module
def test_expand_last_dim_return_tensor_with_more_dims(tmp_path):
    layer = layer_module.ExpandLastDim()
    tensor = layer(tf.constant([0.1, 0.2], dtype=tf.float32))
    assert 2 == len(tensor.shape.as_list())