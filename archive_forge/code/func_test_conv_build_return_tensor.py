import keras_tuner
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_conv_build_return_tensor():
    block = blocks.ConvBlock()
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 32, 3), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1