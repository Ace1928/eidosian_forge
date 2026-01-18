import keras_tuner
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_conv_with_small_image_size_return_tensor():
    block = blocks.ConvBlock()
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(10, 10, 3), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1